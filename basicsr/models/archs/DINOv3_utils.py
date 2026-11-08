import torch
import torch.nn as nn
import torch.nn.functional as F
import re


REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12, "VITSP": 12, "VITB": 12,
    "VITL": 24, "VITHP": 32, "VIT7B": 40,
}


class DINOExtractor(nn.Module):

    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=(5, 11, 17, 23),    # Zero-based indices. For DINOv3 ViT-L/16 with 24 blocks, these correspond to the 6th, 12th, 18th, and 24th layers.
        input_size=512,
        device=None,
    ):
        super().__init__()

        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        ).eval()

        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = list(extract_ids)
        self.input_size = int(input_size)
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f'DINO input_size ({self.input_size}) must be divisible by '
                f'patch_size ({self.patch_size}).'
            )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input image tensor with shape [B, 3, H, W], expected to be in [0, 1].

        Returns:
            A list of DINOv3 feature maps extracted from selected intermediate layers.
            For DINOv3 ViT-L/16 with 512x512 input, each feature map has shape [B, 1024, 32, 32].
        """
        x = x.clamp(0, 1)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.mean) / self.std

        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                x, n=self.extract_ids, reshape=True, norm=True
            )

        return list(feats)


class DimAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                reduction, reduction, kernel_size=3, padding=1,
                groups=reduction, bias=False
            ),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv2d(reduction, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        return self.proj(x)


class SemanticAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, sizes=(32, 16, 8, 4), bottleneck=64):
        super().__init__()
        self.sizes = list(sizes)
        self.blocks = nn.ModuleList([
            DimAdapter(in_dim, out_dim, reduction=bottleneck)
            for _ in self.sizes
        ])

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            outs.append(self.blocks[i](x))
        return outs


class SemanticModulator(nn.Module):
    def __init__(self, dim, condition_dim=256):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Conv2d(condition_dim, dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 1),
        )
        self.modulation_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, condition):
        if condition.shape[2:] != x.shape[2:]:
            condition = F.interpolate(
                condition, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        scale, shift = self.condition_proj(condition).chunk(2, dim=1)
        x_sft = x * (1 + scale) + shift
        return x + self.modulation_strength * (x_sft - x)


class DIRFeatureRouter(nn.Module):
    """Degradation-aware information routing over adapted DINOv3 features.

    Four adapted DINOv3 layers are treated as experts. The expert-level router
    (ER) uses global U-Net features and the degradation vector, while the
    region-level router (RR) uses spatial U-Net features. Private and shared
    priors are fused by a lightweight gate before residual SFT injection.
    """

    def __init__(
        self,
        dim,
        condition_dim=256,
        degradation_condition_dim=None,
        degradation_dim=512,
        num_experts=4,
        router_hidden=None,
        routing_type="er_rr",
        gate_init=0.0,
        temperature=1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.degradation_dim = degradation_dim
        self.routing_type = routing_type.lower()
        self.temperature = max(float(temperature), 1e-6)

        if self.routing_type not in {"er", "rr", "er_rr", "uniform"}:
            raise ValueError(f"Unsupported dir routing_type: {routing_type}")

        router_hidden = router_hidden or max(dim // 4, 16)
        degradation_condition_dim = degradation_condition_dim or condition_dim

        self.private_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(condition_dim, dim, 1, bias=True),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_experts)
        ])
        self.shared_proj = nn.Sequential(
            nn.Conv2d(condition_dim, dim, 1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.degradation_prior_proj = nn.Sequential(
            nn.Conv2d(degradation_condition_dim, dim, 1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.damage_prior_proj = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.prior_context_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
        )
        self.routing_prior_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.degradation_prior_gate = nn.Parameter(torch.tensor(float(gate_init)))

        if self.routing_type in {"er", "er_rr"}:
            self.er_x_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, router_hidden, 1, bias=True),
            )
            self.er_deg_proj = nn.Linear(degradation_dim, router_hidden, bias=True)
            self.er_out = nn.Sequential(
                nn.SiLU(inplace=True),
                nn.Conv2d(router_hidden, num_experts, 1, bias=True),
            )
        else:
            self.er_x_proj = None
            self.er_deg_proj = None
            self.er_out = None

        if self.routing_type in {"rr", "er_rr"}:
            self.rr_router = nn.Sequential(
                nn.Conv2d(dim * 3, router_hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(router_hidden, num_experts, 3, padding=1, bias=True),
            )
        else:
            self.rr_router = None

        self.private_shared_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.prior_to_sft = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 1, bias=True),
        )
        self.dir_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _resize_condition(self, condition, size):
        if condition.shape[2:] != size:
            condition = F.interpolate(
                condition, size=size, mode="bilinear", align_corners=False
            )
        return condition

    def forward(
        self,
        x,
        conditions,
        degradation_vector=None,
        degradation_condition=None,
        damage_map=None,
    ):
        b, _, h, w = x.shape
        if isinstance(conditions, dict):
            size = (h, w)
            conditions = conditions.get(size, conditions.get("base"))
            if conditions is None:
                raise ValueError(f"No precomputed DINO features for size {size}.")

        if not isinstance(conditions, (list, tuple)):
            conditions = [conditions]
        if len(conditions) < self.num_experts:
            raise ValueError(
                f"DIRFeatureRouter needs {self.num_experts} DINO features, "
                f"but got {len(conditions)}."
            )

        private_feats = []
        shared_feats = []
        for condition, private_proj in zip(conditions[:self.num_experts], self.private_projs):
            condition = self._resize_condition(condition, (h, w))
            private_feats.append(private_proj(condition))
            shared_feats.append(self.shared_proj(condition))

        private_stack = torch.stack(private_feats, dim=1)
        shared_prior = torch.stack(shared_feats, dim=1).mean(dim=1)

        if degradation_condition is None:
            degradation_prior = x.new_zeros(b, x.shape[1], h, w)
        else:
            degradation_condition = self._resize_condition(
                degradation_condition, (h, w)
            )
            degradation_prior = self.degradation_prior_proj(degradation_condition)

        if damage_map is None:
            damage_prior = x.new_zeros(b, x.shape[1], h, w)
        else:
            damage_map = self._resize_condition(damage_map, (h, w))
            damage_prior = self.damage_prior_proj(damage_map)

        degradation_damage_prior = self.prior_context_proj(
            torch.cat([degradation_prior, damage_prior], dim=1)
        )
        routing_context = x + torch.tanh(self.routing_prior_gate) * degradation_damage_prior

        if self.routing_type == "uniform":
            route_weight = x.new_full(
                (b, self.num_experts, 1, h, w),
                1.0 / self.num_experts,
            )
        else:
            route_logits = None
            if self.er_x_proj is not None:
                if degradation_vector is None:
                    degradation_vector = x.new_zeros(b, self.degradation_dim)
                er_feat = self.er_x_proj(routing_context)
                er_deg = self.er_deg_proj(degradation_vector).view(b, -1, 1, 1)
                route_logits = self.er_out(er_feat + er_deg)
            if self.rr_router is not None:
                rr_logits = self.rr_router(
                    torch.cat([x, degradation_prior, damage_prior], dim=1)
                )
                route_logits = rr_logits if route_logits is None else route_logits + rr_logits
            route_weight = F.softmax(route_logits / self.temperature, dim=1).unsqueeze(2)

        private_prior = (route_weight * private_stack).sum(dim=1)
        prior_gate = self.private_shared_gate(
            torch.cat([routing_context, private_prior, shared_prior], dim=1)
        )
        fused_prior = prior_gate * private_prior + (1 - prior_gate) * shared_prior
        fused_prior = fused_prior + torch.tanh(self.degradation_prior_gate) * degradation_damage_prior
        scale, shift = self.prior_to_sft(fused_prior).chunk(2, dim=1)
        x_sft = x * (1 + scale) + shift
        return x + torch.tanh(self.dir_gate) * (x_sft - x)
