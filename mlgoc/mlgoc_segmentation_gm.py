import numpy as np
import math
import matplotlib.pyplot as plt

def mlgoc_segmentation_gm(parameters_pf,
                          extended_image,
                          data_parameters,
                          kappa,
                          extended_initial_phi,
                          optimization_parameters):
    """ Main function for multi-layered 'gas of circles' segmentation
    
    :param parameters_pf: 
    :param extended_image: 
    :param data_parameters: 
    :param kappa: 
    :param extended_initial_phi: 
    :param optimization_parameters: 
    :return: 
    """

    extended_image_size = np.shape(extended_image)
    extended_image_height = extended_image_size[0]
    extended_image_width = extended_image_size[1]


    maxd = int(max(map(lambda x:x['d'], parameters_pf)))
    image_height = extended_image_height - 4*maxd
    image_width = extended_image_width - 4*maxd

    f_phi = ml_evolution(extended_initial_phi,
                         kappa,
                         optimization_parameters['tolerance'],
                         optimization_parameters['max_iterations'],
                         optimization_parameters['save_frequency'],
                         parameters_pf,
                         data_parameters,
                         extended_image)
    if f_phi.ndim > 2:
        final_phi = f_phi[:,2*maxd:-2*maxd,2*maxd:-2*maxd]
    else:
        final_phi = f_phi[2*maxd:-2*maxd,2*maxd:-2*maxd]
    return final_phi


def ml_evolution(init_phi,
                 kappa,
                 tolerance,
                 max_iterations,
                 save_frequency,
                 parameters,
                 data_parameters,
                 ext_image):
    """ Gradient descent algorithm for phase field optimization
    
    :param init_phi: 
    :param kappa: weight of overlap penalty (real number)
    :param tolerance: 
    :param max_iterations: 
    :param save_frequency: 
    :param parameters: 
    :param data_parameters: 
    :param ext_image: padded input image
    :return: 
    """
    linear_operator = compute_linear_part(init_phi, parameters)
    old_phi = init_phi
    converged = False
    num_of_iterations = 0
    new_phi = init_phi

    while not converged:

        functional_derivative, overlap_derivative = ml_evolve_step(old_phi,
                                                                   linear_operator,
                                                                   parameters,
                                                                   data_parameters,
                                                                   ext_image,
                                                                   kappa)
        functional_derivative = functional_derivative + overlap_derivative
        max_functional_derivative = np.max(np.abs(functional_derivative))
        delta_t = 1.0/(10.0*max_functional_derivative)
        delta_phi = -delta_t * functional_derivative
        new_phi = old_phi + delta_phi
        mean_functional_derivative = np.mean(np.abs(functional_derivative ))

        if max_iterations<10:
            print('Iteration {0:6d} ({1:3d}%)'.format(num_of_iterations, int((100.0 * num_of_iterations) / max_iterations)))
        elif num_of_iterations % int(max_iterations/10) == 0:
            print('Iteration {0:6d} ({1:3d}%)'.format(num_of_iterations, int((100.0*num_of_iterations)/max_iterations)))

        old_phi = new_phi
        num_of_iterations = num_of_iterations + 1
        if mean_functional_derivative < tolerance or num_of_iterations >= max_iterations:
            converged = True
            print('Iteration {0:6d} ({1:3d}%)'.format(max_iterations, 100 ))
    return new_phi


def ml_evolve_step(old_phi,
                   linear_operator,
                   parameters,
                   data_parameters,
                   image,
                   kappa):
    """ Computes the functional derivative of the multi-layered phase field
    
    :param old_phi: 
    :param linear_operator: 
    :param parameters: 
    :param data_parameters: 
    :param image: padded input image
    :param kappa: weight of overlap penalty (real number)
    :return: 
    """
    phase_size = old_phi.shape
    if old_phi.ndim > 2:
        layer_number = phase_size[0]
    else:
        layer_number = 1
    kappas = [kappa]*layer_number

    functional_derivative = np.zeros(phase_size)
    overlap_derivative = np.zeros(phase_size)

    image_part = compute_imagepart_additivetanh(old_phi, image, data_parameters)

    if layer_number > 1:
        sum_phi = np.sum(old_phi, axis=0)
        for i in range(layer_number):
            hat_old_phi = np.fft.fftshift(np.fft.fft2(old_phi[i, :, :]))
            op_old_phi = linear_operator[i] * hat_old_phi
            linear_part = np.real( np.fft.ifft2( np.fft.ifftshift( op_old_phi ) ))
            nonlinear_part = compute_nonlinear_part(old_phi[i,:,:], parameters[i]['lambda'], parameters[i]['alpha'])
            functional_derivative[i,:,:] = linear_part + parameters[i]['alpha'] + nonlinear_part + image_part[i, :, :]

            overlap_derivative[i,:,:] = kappas[i]/2.0 * (sum_phi - old_phi[i,:,:] + layer_number - 1)
    else:
        hat_old_phi = np.fft.fftshift(np.fft.fft2(old_phi))
        op_old_phi = linear_operator[0] * hat_old_phi
        linear_part = np.real(np.fft.ifft2(np.fft.ifftshift(op_old_phi)))
        nonlinear_part = compute_nonlinear_part(old_phi, parameters[0]['lambda'], parameters[0]['alpha'])
        functional_derivative = linear_part + parameters[0]['alpha'] + nonlinear_part + image_part
    return functional_derivative, overlap_derivative


def compute_linear_part(init_phi, parameters):

    layer_number = init_phi.shape[0]

    linear_operator = [None] * layer_number
    for i in range(layer_number):
        temp_params = parameters[i]
        D_pf = temp_params['D']
        lambda_pf = temp_params['lambda']
        beta_pf = temp_params['beta']
        ny = init_phi.shape[1]
        nx = init_phi.shape[2]
        k2 = compute_neg_laplacian(ny, nx, temp_params['discrete'])
        interaction_operator = compute_interaction_operator(k2, temp_params)
        linear_operator[i] = k2 * (D_pf - beta_pf * interaction_operator) - lambda_pf
    return linear_operator


def compute_nonlinear_part(phi, lambda_pf, alpha_pf):
    phi2 = phi*phi
    phi3 = phi2*phi
    return lambda_pf*phi3 - alpha_pf*phi2


def compute_neg_laplacian(height, width, discrete):
    x = np.arange(-math.pi, math.pi, 2 * math.pi / width) * (width - 1) / width
    y = np.arange(-math.pi, math.pi, 2 * math.pi / height) * (height - 1) / height
    kx, ky = np.meshgrid(x,y)
    if discrete:
        k2 = 2*(2 - np.cos(kx) - np.cos(ky))
    else:
        k2 = kx*kx + ky*ky
    return k2


def compute_interaction_operator(k2, parameters):
    """ Computes the interaction operator in Fourier space
    
    :param k2: negative Laplacian operator, used to get the operator size
    :param parameters: dictionary that contains the keys 'd' and 'epsilon' to control interaction range
    :return: 
    """
    if not parameters['marie']:
        return 1.0 / (1.0 + k2)
    else:
        k2_size = np.shape(k2)
        height = k2_size[0]
        width = k2_size[1]
        d = parameters['d']
        epsilon = parameters['epsilon']
        x = np.arange(-width/2.0,width/2,1.0)
        y = np.arange(-height/2.0,height/2,1.0)
        kx, ky = np.meshgrid(x, y)
        r = np.sqrt(kx*kx + ky*ky)
        inner_space_interaction_operator = r <= (d - epsilon)
        centred_r = (r-d) / epsilon
        centre_space_interaction_operator = 0.5 * (1.0 - centred_r - np.sin(math.pi*centred_r) / math.pi ) * (r > (d-epsilon)) * (r < (d+epsilon) )
        space_interaction_operator = inner_space_interaction_operator + centre_space_interaction_operator
        return np.real( np.fft.fftshift( np.fft.fft2( np.fft.ifftshift(space_interaction_operator) ) ) )


def compute_imagepart_additivetanh(phi, image, data_parameters):
    """ Computes the functional derivative of the additive image term
    
    :param phi: HxWXL sized matrix, the multi-layered phase field, where H and W are the height and width of the image,
            and L is the number of layers
    :param image: HxW sized matrix, the input image
    :param data_parameters: a dictionary that contains the parameters of the data term 
            including the keys {'muin', 'sigmain', 'muout', 'sigmaout', 'gamma1', 'gamma2'}
    :return: 
    """
    tanh_phi = np.tanh(phi)
    phase_size = phi.shape
    if phi.ndim > 2:
        layer_number = phase_size[0]
        tilde_phi_plus = np.sum(tanh_phi, axis=0) + layer_number/2
    else:
        layer_number = 1
        tilde_phi_plus = tanh_phi + layer_number/2

    tilde_phi_plus2 = tilde_phi_plus * tilde_phi_plus
    sech2_phi = np.power(1. / np.cosh(phi), 2)
    muin = data_parameters['muin']
    sigmain = data_parameters['sigmain']
    muout = data_parameters['muout']
    sigmaout = data_parameters['sigmaout']
    deltamu = muin - muout
    deltamu2 = deltamu * deltamu
    sigmaout2 = sigmaout * sigmaout
    deltasigma2 = sigmain * sigmain - sigmaout2
    image_minus_muout = image - muout

    intensity_part_nominator = deltamu2 * deltasigma2 * tilde_phi_plus2 \
                               + 2 * sigmaout2 * deltamu2 * tilde_phi_plus \
                               - 2 * sigmaout2 * deltamu * image_minus_muout \
                               - deltasigma2 * image_minus_muout * image_minus_muout
    intensity_part_denominator = (sigmaout2 + deltasigma2 * tilde_phi_plus2) * (
    sigmaout2 + deltasigma2 * tilde_phi_plus2)
    intensity_part_common = intensity_part_nominator / intensity_part_denominator
    image_linear_part = np.zeros(phase_size)

    if layer_number == 1:
        image_linear_part = intensity_part_common * sech2_phi
    else:
        for i in range(layer_number):
            image_linear_part[i, :, :,] = intensity_part_common * sech2_phi[i, :, :]
    image_linear_part *= data_parameters['gamma2'] / 4

    return image_linear_part
