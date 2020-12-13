import torch


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def random_gaussian_attack(image, epsilon):
    gaussian = torch.normal(torch.zeros_like(image), torch.ones_like(image))
    perturbed_image = image + epsilon * gaussian
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image