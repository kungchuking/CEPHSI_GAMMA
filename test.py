import torch
from dataset import dataset
from cep_system import cep_system
from cep_enc import cep_enc
import loss
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os
from thop import profile

if __name__ == "__main__":
    # -- Essential directories (Create them if they don't already exist)
    ess_dirs = ["./test_results/"]
    
    for d in ess_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # -- Some global parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    patch_size = [512, 512]
    n_subexp   = 16         # -- Number of subexposure per frame
    
    # -- The Model
    model = cep_system(sigma_range=[0, 1e-12],
                       ce_code_n=n_subexp,
                       frame_n=n_subexp,
                       opt_cecode=True,
                       n_cam=2,
                       in_channels=1,
                       out_channels=1,
                       patch_size=patch_size).to(device)

    # -- Encoder
    # -- Load ce_code
    ce_code_init = torch.load("checkpoints/epoch_449_valid_psnr_25.609192422507874_train_psnr_27.363531529310823.pth", weights_only=True, map_location=device)["cep_enc.ce_weight"]
    cep_enc = cep_enc(
            sigma_range=[0, 1e-12],
            ce_code_n=n_subexp,
            frame_n=n_subexp,
            ce_code_init=ce_code_init,
            opt_cecode=False,
            patch_size=patch_size,
            in_channels=1,
            n_cam=2).to(device)

    # -- Load the checkpoint
    model.load_state_dict(torch.load("checkpoints/epoch_331_valid_psnr_25.200943279902496_train_psnr_27.06679347584457.pth", weights_only=True, map_location=device))

    # -- Load the checkpoint to just the encoder.
    # -- cep_enc.load_state_dict(torch.load("checkpoints/epoch_331_valid_psnr_25.200943279902496_train_psnr_27.06679347584457.pth", weights_only=True, map_location=device)["cep_enc.ce_weight"])

    model.eval()
    cep_enc.eval()

    params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 1
            }
    
    test_set = dataset(ds_dir="./dataset/custom/", n_subframe=n_subexp, patch_size=patch_size)
    test_gen = torch.utils.data.DataLoader(test_set, **params)
    
    print ("[INFO] len(test_set): ", len(test_set))

    # -- Testging
    # -- Initialize the running sum of some relevant metrics (i.e. PSNR)
    run_psnr_avg = .0
    with torch.set_grad_enabled(False):
        for i, data in tqdm(enumerate(test_gen)):
            data = data.to(device).float() / 255.
    
            # -- Make predictions
            # -- target == data (i.e. autoencoder)

            _, _, cep_encoded = cep_enc(data)
            output, ce_blur, _, target, reblur = model(data)

            if i == 0:
                flops, params = profile(model, inputs=(data,))
                print ("[INFO] flops: {}".format(flops))
                print ("[INFO] params: {}".format(params))
    
            # -- Clamp the output prior to computing the PSNR
            output = torch.clamp(output, min=0., max=1.)
    
            # -- Iterate over each batch and each subframe to calculate the average PSNR of the current batch.
            run_psnr = .0
            for m in range(output.shape[0]):
                for n in range(output.shape[1]):
                    run_psnr += psnr(output[m, n, 0, ...].detach().cpu().numpy(), target[m, n, 0, ...].detach().cpu().numpy())
            psnr_avg = run_psnr / float(output.shape[0] * output.shape[1])
    
            run_psnr_avg += psnr_avg

            # -- print ("[INFO] cep_encoded.shape: ", cep_encoded.shape)
            # -- Save the encoded images for inspection
            for j in range(cep_encoded.shape[1]):
                fig = plt.figure()
                plt.imshow(cep_encoded[0, j, ...].detach().cpu().numpy(), cmap="gray")
                plt.xticks([])                
                plt.yticks([])                
                # fig.patch.set_facecolor("red")
                plt.savefig("./test_results/cep_encoded_i_{}_j_{}.png".format(i, j))
                plt.close(fig)

            # -- Save output images for inspection
            for j in range(output.shape[1]):
                fig = plt.figure()
                plt.imshow(output[0, j, 0, ...].detach().cpu().numpy(), cmap="gray")
                plt.xticks([])                
                plt.yticks([])                
                # fig.patch.set_facecolor("red")
                plt.savefig("./test_results/test_i_{}_j_{}.png".format(i, j))
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(data[0, j, 0, ...].detach().cpu().numpy(), cmap="gray")
                plt.xticks([])                
                plt.yticks([])                
                # fig.patch.set_facecolor("red")
                plt.savefig("./test_results/gt_i_{}_j_{}.png".format(i, j))
                plt.close(fig)
    
        # -- Report the performance on the validation set
        test_psnr = run_psnr_avg / float(i+1)
        print ("[INFO] Test Set PSNR: {};".format(test_psnr))

