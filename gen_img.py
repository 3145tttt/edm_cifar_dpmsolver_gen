import pickle
import dnnlib
import torch

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
# Диффузионка из статьи https://arxiv.org/abs/2206.00364

def load_edm_net(
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
    ):

    with dnnlib.util.open_url(network_pkl, verbose=1) as f:
        net = pickle.load(f)['ema']

    for param in net.parameters():
        param.requires_grad = False

    return net

class VPScheduler:

    def __init__(self):
        beta_min = 0.1
        beta_d   = 19.9

        self.vp_sigma   = lambda t: (torch.exp(0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        self.s          = lambda t: torch.exp(-1./2. * (0.5 * beta_d * (t ** 2) + beta_min * t))
        
        # \hat{\alpha}(t) = \exp{-\int_{0}^t (beta_d s + beta_min) ds } 
        # vp_sigma(t)     = \sqrt{1 /  \hat{\alpha}(t) - 1}
        # s(t)            = \sqrt{\hat{\alpha}(t)}

def predict_x0_VP_SDE(net, scheduler):

    def diff_x0_fn(x, t):

        # https://arxiv.org/abs/2206.00364
        # table 1, column 1
        s = scheduler.s(t)[:, None, None, None]
        sigma = scheduler.vp_sigma(t)[:, None, None, None]

        return net(x / s, sigma)

    return diff_x0_fn

net = load_edm_net()
scheduler = VPScheduler()

diff_x0_fn_edm = predict_x0_VP_SDE(net, scheduler)

# Пример использования
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Параметры
num_steps = 10 # NFE
t_steps = None # Если хочестя задать свои, то надо присвоить
latents = torch.randn((16, 3, 32, 32))


noise_schedule = NoiseScheduleVP('linear') # \beta(t) = 20 t + 0.01
model_fn = model_wrapper(
    diff_x0_fn_edm,
    noise_schedule=noise_schedule,
    model_type="x_start", #"x_start"
)

def custom_get_time_steps(skip_type, t_T, t_0, N, device):
    return t_steps

dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

if t_steps is not None:
    dpm_solver.get_time_steps = custom_get_time_steps

images = dpm_solver.sample(
    latents,
    steps=num_steps,
    order=2,
    skip_type="logSNR",
    method="multistep",
)

# Как сохранять. Очень важно
# source https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py#L297
import os
import PIL.Image

outdir = "./imgs"
already_sampled = len(os.listdir(outdir))

images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
for idx, image_np in enumerate(images_np):
    
    save_idx = idx + 1 + already_sampled
    image_path = os.path.join(outdir, f'{save_idx:06d}.png')
    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
