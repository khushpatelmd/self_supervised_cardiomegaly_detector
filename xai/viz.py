from config import *
from model_bal import *
idxx = 25
directory = "img_25_1"

ckpt = "epoch=117-val_loss=0.92-val_f1=0.80.ckpt"

net = balance_resnet()
net = net.to(device)
checkpoint = torch.load(ckpt, map_location=device)
net.load_state_dict(checkpoint["state_dict"])
test_dataloader = DataLoader(test_dataset_bal, batch_size=1, num_workers=NUM_WORKERS)


images, labels = test_dataset_bal[idxx]
input = torch.squeeze(images)
input = images
input.requires_grad = True    
input = torch.unsqueeze(input, axis=0)
net = net.eval()
input = input.to(device)
labels = labels.to(device)

logits = net(input)
img = np.squeeze(images.cpu().detach().numpy())

plt.imsave(directory+"original_image.jpeg", img, cmap="gray")

img = np.expand_dims(img, axis=-1)

print("Predicted label",torch.argmax(logits))
print("original label",labels)

def attribute_image_features(algorithm, input, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels,
                                              **kwargs
                                             )
    
    return tensor_attributions
  
saliency = Saliency(net)
grads = saliency.attribute(input, target=labels.item())
grads = np.squeeze(grads.cpu().detach().numpy())


grad = np.expand_dims(grads, axis=-1)

fig, _ = viz.visualize_image_attr(grad, img, method="blended_heat_map",sign="all",
                          show_colorbar=True, title="Saliency", alpha_overlay=0.7)
fig.savefig(directory+"Saliency.jpeg")

ig = IntegratedGradients(net)
attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
# attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
print('Approximation delta: ', abs(delta))
attr_ig = np.squeeze(attr_ig.cpu().detach().numpy())
attr_ig = np.expand_dims(attr_ig, axis=-1)
fig, _ = viz.visualize_image_attr(attr_ig, img, method="blended_heat_map",sign="all",
                          show_colorbar=True, title="Integrated Gradients")

fig.savefig(directory+"Integrated_Gradients.jpeg")

gradient_shap = GradientShap(net)
# Defining baseline distribution of images
rand_img_dist = torch.cat([input * 0, input * 1])

attributions_gs = gradient_shap.attribute(input,
                                          n_samples=10,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=labels)

attributions_gs = np.squeeze(attributions_gs.cpu().detach().numpy())
attributions_gs = np.expand_dims(attributions_gs, axis=-1)

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

fig, _ = viz.visualize_image_attr(attributions_gs, img, method="blended_heat_map",sign="all",
                          show_colorbar=True, title="GradientShap")

fig.savefig(directory+"Gradient_Shap.jpeg")

occlusion = Occlusion(net)

attributions_occ = occlusion.attribute(input,
                                       strides = (1, 8, 8),
                                       target=labels,
                                       sliding_window_shapes=(1,15, 15),
                                       baselines=0)
attributions_occ = np.squeeze(attributions_occ.cpu().detach().numpy())
attributions_occ = np.expand_dims(attributions_occ, axis=-1)

figure, axis = viz.visualize_image_attr(attributions_occ, img, method="blended_heat_map",sign="all",
                          show_colorbar=True, title="Overlayed occlusion")
figure.savefig(directory+"occlusion.jpeg")

guided_gc = GuidedGradCam(net, net.model.layer4[2].conv3)
attribution = guided_gc.attribute(input, 0)
attribution = np.squeeze(attribution.cpu().detach().numpy())
attribution = np.expand_dims(attribution, axis=-1)
figure, = viz.visualize_image_attr(attribution, img, method="blended_heat_map", sign="all",
                          show_colorbar=True, title="Overlayed Gradient Magnitudes")
figure.savefig(directory+"guided_grad_cam.jpeg")