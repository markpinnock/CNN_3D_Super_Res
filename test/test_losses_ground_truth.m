FILE_PATH = "C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\CNN_3D_Super_Res\scripts\test\";

img1 = zeros(64, 64);
img2 = ones(64, 64);
img3 = zeros(64, 64);

img1(17:48, 17:48) = 1;
img2 = img2 - img1;
img3(1:8:64, 1:8:64) = 1;

% subplot(1, 2, 1), imshow(img1);
% subplot(1, 2, 2), imshow(img3);

% vol1 = repmat(img1, [1 1 4]);
% vol2 = repmat(img1, [1 1 4]);
% loss = sum((vol1 - vol2).^2, [1 2])
% 
% vol1 = repmat(img1, [1 1 4]);
% vol2 = repmat(img2, [1 1 4]);
% loss = sum((vol1 - vol2).^2, [1 2])

fft_img1 = fft(img1(:) - mean(img1(:)));
fft_img2 = fft(img2(:) - mean(img2(:)));
fft_img3 = fft(img3(:) - mean(img3(:)));

N = length(img1(:))/2;
k = linspace(0, 1, N);

subplot(1, 3, 1), plot(k, abs(fft_img1(1:N)));
subplot(1, 3, 2), plot(k, abs(fft_img2(1:N)));
subplot(1, 3, 3), plot(k, abs(fft_img3(1:N)));



%%
N = 1000;
tempx = linspace(0, 4*pi, N);
deltax = tempx(2) - tempx(1);
kmax = 1 / deltax;
tempy1 = cos(pi*tempx);
tempy2 = cos(pi*tempx) + cos(2*pi*tempx);
tempy3 = cos(pi*tempx) + cos(2*pi*tempx) + cos(3*pi*tempx);
fftx = linspace(0, 2*kmax, N);
ffty1 = fft(tempy1);
ffty2 = fft(tempy2);
ffty3 = fft(tempy3);
subplot(1, 3, 1), plot(fftx(1:N/2), abs(ffty1(1:N/2)));
subplot(1, 3, 2), plot(fftx(1:N/2), abs(ffty2(1:N/2)));
subplot(1, 3, 3), plot(fftx(1:N/2), abs(ffty3(1:N/2)));


