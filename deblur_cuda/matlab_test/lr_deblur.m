clc; clear; close all;

% a lot was taken from this Stack Overflow answer
% https://stackoverflow.com/a/35259180

%%
original = im2double(imread('earth_blurry.png'));

% original = original(:,:,3);
original = original + 0.1;

figure; imshow(original); title('Original Image')

%%
sigma=2;
fltSize=round(sigma*6)+1;
x = -floor(fltSize/2):(-floor(fltSize/2)+fltSize-1);
PSF = exp(-(x.^2+x.'.^2)/(2*sigma^2))/(2*pi*sigma^2);
figure()
imshow(PSF);

% PSF = fspecial('gaussian', hsize, sigma);
% figure();
% imshow(PSF);

% blr = imfilter(original, PSF);
blr = original;
figure; imshow(blr); title('Blurred Image')

%%

iter = 1;
res_RL = 0.5*ones(size(original));
res_RL(:,:,1) = RL_deconv(blr(:,:,1), PSF, iter); 
res_RL(:,:,2) = RL_deconv(blr(:,:,2), PSF, iter); 
res_RL(:,:,3) = RL_deconv(blr(:,:,3), PSF, iter);

figure; imshow(res_RL); title('Recovered Image')

imwrite(res_RL, '~/test2.png')

%%

function result = RL_deconv(image, PSF, iterations)
    % to utilise the conv2 function we must make sure the inputs are double
    image = double(image);
    PSF = double(PSF);
    latent_est = 0.5*ones(size(image));%image; % initial estimate, or 0.5*ones(size(image)); 
    PSF_HAT = PSF(end:-1:1,end:-1:1); % spatially reversed psf
    % iterate towards ML estimate for the latent image
    for i= 1:iterations
        fprintf("iteration %d\n", i);
        est_conv      = conv2(latent_est,PSF,'same');
        relative_blur = image./est_conv;
        error_est     = conv2(relative_blur,PSF_HAT,'same');
        latent_est    = latent_est.* error_est;
        
%         latent_est = relative_blur;
    end
    result = latent_est;
end