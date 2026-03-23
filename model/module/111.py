class VRWKV_SpatialMix_wkv5(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        attn_bias=False,
        init_mode="fancy",
        key_norm=True,
        with_cls_token=False,
        with_cp=False,
        img_txt_cat_order=0,
        scan_K=1,
        head_div=1,
        decay_speed="slow",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.with_cp = with_cp
        self.img_txt_cat_order = img_txt_cat_order
        self.scan_K = scan_K
        self.n_head = n_head
        head_size = n_embd // n_head
        assert head_size * n_head == n_embd, "n_embd should be divisible by n_head"
        self.head_size = head_size
        self.head_size_divisor = head_div

        # shift
        # self.omni_shift = OmniShift(dim=n_embd)

        # key, value, receptance
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = partial(F.normalize, dim=-1)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, attn_sz, bias=False)
        # self.ln_x = nn.GroupNorm(n_head, attn_sz)
        self.ln_x = nn.LayerNorm(attn_sz)

        # spatial decay and spatial first
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            # self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            # self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            # self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for n in range(attn_sz):
                if decay_speed == "slow":
                    decay_speed[n] = -6 + 5 * (n / (attn_sz - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )
                else:
                    decay_speed[n] = -0.3 + 1 * (n / (attn_sz - 1)) ** (
                        0.1 + 1.3 * ratio_0_to_1
                    )

            self.time_decay = nn.Parameter(
                decay_speed.reshape(self.n_head, self.head_size)
            )
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(attn_sz)
            for n in range(attn_sz):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (attn_sz - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    # def shift_scanned_x(self, x, h, w):
    #     assert self.scan_K in [0, 2], "scan_K should be 0 or 2"
    #     if self.scan_K == 2:
    #         xx = rearrange(x, '(b k) l c -> b k c l', h=h, w=w, k=self.scan_K)
    #         # same
    #         xx_0 = rearrange(xx[:, 0], 'b c (h w) -> b c h w', h=h, w=w)
    #         xx_0 = self.omni_shift(xx_0)
    #         xx_0 = rearrange(xx_0, 'b c h w -> b c (h w)')
    #         # transposed
    #         xx_1 = rearrange(xx[:, 1], 'b c (w h) -> b c h w', h=h, w=w)
    #         xx_1 = self.omni_shift(xx_1)
    #         xx_1 = rearrange(xx_1, 'b c h w -> b c (h w)')

    #         xx = torch.stack([xx_0, xx_1], dim=1)

    #         xx = rearrange(xx, 'b k c h w -> b c h w')
    #         return xx

    def jit_func(
        self,
        x,
        txt=None,
        resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        h, w = resolution

        # xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # xx = self.omni_shift(xx)
        # xx = rearrange(xx, 'b c h w -> b (h w) c')

        # xx = x

        # cat txt and special tokens
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                # xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(1) + 4
            else:
                x = self.cat_seqs_wo_special_tokens(x, txt)
                # xx = self.cat_seqs_wo_special_tokens(xx, txt)
                T = T + txt.size(1)

        # time decay
        # xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        # xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        # xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        # xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # sr = torch.sigmoid(r)
        g = F.silu(self.gate(x))

        return r, k, v, g, T

    def jit_func_2(self, x, g):
        # B, T, C = x.size()
        # x = x.reshape(B * T, C)
        # x = self.ln_x(x / self.head_size_divisor).reshape(B, T, C)

        x = self.ln_x(x)
        x = self.output(x * g)
        return x

    def forward(
        self,
        x,
        txt=None,
        patch_resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        def _inner_forward(x, txt, patch_resolution, mm_tokens):
            B, T, C = x.size()
            # sr, k, v, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            r, k, v, g, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            # x = RUN_CUDA_RWKV5(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            x = RUN_CUDA_RWKV5_2(
                B, T, C, self.n_head, r, k, v, w=self.time_decay, u=self.time_faaaa
            )
            # print(f'Spatial mix - x max: {x.abs().max()}, x norm: {x.norm()}')
            x = self.key_norm(x)
            # x = sr * x
            # x = self.output(x)
            x = self.jit_func_2(x, g)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                _inner_forward, x, txt, patch_resolution, mm_tokens, use_reentrant=False
            )
        else:
            x = _inner_forward(x, txt, patch_resolution, mm_tokens)
        return x

