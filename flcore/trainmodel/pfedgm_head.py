import torch
import torch.nn as nn

class NavigatorHead(nn.Module):

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.mu = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.normal_(self.mu, std=0.01)
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.log_A = nn.Parameter(torch.zeros(num_classes, in_features))

    def forward(self, z):
        return z @ self.mu.t() - 0.5 * (self.mu * self.mu).sum(1) + self.b

    def cov_logits(self, z_detached):
        A = self.log_A.exp()
        mu = self.mu.detach()
        z2 = z_detached * z_detached @ A.t()
        zm = z_detached @ (A * mu).t()
        mm = (A * mu * mu).sum(1)
        return -0.5 * (z2 - 2.0 * zm + mm) + self.b.detach()

class PGMFusionHead(nn.Module):

    def __init__(self, in_features, num_classes, lam, variant='drop'):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.lam = float(lam)
        self.variant = variant
        self.mu_off = nn.Parameter(torch.zeros(num_classes, in_features))
        self.b_off = nn.Parameter(torch.zeros(num_classes))
        self.log_Ag = nn.Parameter(torch.zeros(in_features))
        self.log_Ac = nn.Parameter(torch.zeros(in_features))
        self.register_buffer('mu_star', torch.zeros(num_classes, in_features))
        self.register_buffer('b_star', torch.zeros(num_classes))
        self.register_buffer('A_star', torch.ones(num_classes, in_features))
        self.register_buffer('upsilon', torch.zeros(num_classes, in_features))
        self.register_buffer('present', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def set_globals(self, mu_star, b_star, A_star, upsilon, present):
        self.mu_star.copy_(mu_star)
        self.b_star.copy_(b_star)
        self.A_star.copy_(A_star)
        self.present.copy_(present)
        ups = upsilon.clone()
        if self.variant == 'global':
            absent = ~present
            ups[absent] = mu_star[absent]
        self.upsilon.copy_(ups)

    def forward(self, z):
        d = self.in_features
        Ag = self.log_Ag.exp()
        Ac = self.log_Ac.exp()
        mu_g = self.mu_star + self.mu_off
        W = Ag.unsqueeze(0) * self.A_star
        z2 = z * z @ W.t()
        zm = z @ (W * mu_g).t()
        mm = (W * mu_g * mu_g).sum(1)
        xi = -0.5 * (z2 - 2.0 * zm + mm)
        t1 = z * z @ Ac.unsqueeze(1)
        t2 = z @ (Ac.unsqueeze(0) * self.upsilon).t()
        t3 = (Ac.unsqueeze(0) * self.upsilon * self.upsilon).sum(1)
        zeta = -(self.lam / d) * (t1 - 2.0 * t2 + t3.unsqueeze(0))
        if self.variant != 'global':
            zeta = zeta * self.present.to(zeta.dtype).unsqueeze(0)
        return xi + zeta + (self.b_star + self.b_off).unsqueeze(0)
