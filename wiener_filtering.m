%--------------------------------------------------------------------------
% q1
%--------------------------------------------------------------------------
close all; clear all; clc
nfft = 512;
% Read image
x = imread("F16.tiff");
xYcbcr = rgb2ycbcr(x);
y = double(xYcbcr(:, :, 1));

% Add noise to Y channel
noiseVar = SNR2var(10, y);
noise = sqrt(noiseVar) * randn(size(y));
% noisedY = imnoise(y, "gaussian", 0, noiseVar/(255^2));
noisedY = noise + y; 

snrVal = snr(y(:), noise(:));

noisyYcbcr = xYcbcr;
noisyYcbcr(:, :, 1) = noisedY;
noisyRGBx = ycbcr2rgb(noisyYcbcr);

origArray = {x, uint8(y), uint8(noisedY)};
figure("Name", "Original image");
montage(origArray, "Size", [1 3]);
title("Original Image (left), Original Y Channel (middle), Noisy Y Channel (right)");

% Choosing [20, 424] to [94, 498] area to calculate sample variance 
% of noise.
flatPatch = noisedY(424:498, 20:94);

% Estimated noise variance calculated
estNoiseVar = mean((flatPatch - mean(flatPatch, "all")).^2, "all");

% Displaying the patch used
figure("Name", "Flat Patch");
imshow(uint8(flatPatch)); title("Flat Patch to calculate noise variance");

% Taking FFT of the noisy image (function can be found below the script)
FNoisy = twodFFT(noisedY, nfft);

% Power spectrum of the original image calculated
size1 = size(x,1);
size2 = size(x,2);
% SpecYOrig = (1 / (nfft^2)) * abs(FNoisy).^2;
SpecYOrig = (1 / (size1*size2)) * (abs(FNoisy).^2);

% IIR Wiener Filter formed
IIRWiener = SpecYOrig ./ (SpecYOrig + estNoiseVar);

iirWiener = twodIFFT(IIRWiener, nfft, size(y,1)/2-1);
fft_plot(iirWiener, "Wiener Filter", nfft);

% Filtering performed
FOutIIR = IIRWiener .* FNoisy;
% Inverse FFT taken (function can be found below the script)
yIIRWienerOut = twodIFFT(FOutIIR, nfft, size(y,1));

% Transforming back to RGB
YCbCrIIRWienerOut = xYcbcr;
YCbCrIIRWienerOut(:, :, 1) = yIIRWienerOut;
RGBIIRWienerOut = ycbcr2rgb(YCbCrIIRWienerOut);

WienerImgArray = {x, noisyRGBx, RGBIIRWienerOut};

figure("Name", "Wiener Filter Output");
montage(WienerImgArray, "Size", [1 3]); 
title("Original Image (left), Noisy Image (middle), Output of IIR Wiener Filter (right)");

%--------------------------------------------------------------------------
% q2
%--------------------------------------------------------------------------
% Window size = 5
M = floor(5/2);

% Loop to calculate image
filteredPartsx = 1+M:size(y,1)-M;
filteredPartsy = 1+M:size(y,2)-M;

for n1=filteredPartsx
    for n2=filteredPartsy
        locMean = localMean(noisedY, n1, n2, M);
        locVar = localVar(noisedY, n1, n2, M, locMean);
        origImgVar = max(0, locVar - estNoiseVar);
        
        yOut(n1, n2) = locMean + (origImgVar / locVar) *...
            (noisedY(n1, n2) - locMean);
    end
end
% Since I could not find a way to apply filter to the edges of the image, I
% took those parts directly (placing output image to the center part)
yAdaptiveOut = noisedY;
yAdaptiveOut(filteredPartsx, filteredPartsy) = yOut(1+M:end,1+M:end);

% Transforming back to RGB
YCbCrAdaptiveOut = xYcbcr;
YCbCrAdaptiveOut(:, :, 1) = yAdaptiveOut;
RGBAdaptiveOut = ycbcr2rgb(YCbCrAdaptiveOut);

AdaptiveImgArray = {x, noisyRGBx, RGBAdaptiveOut};

figure("Name", "Adaptive Wiener Filter Output");
montage(AdaptiveImgArray, "Size", [1 3]); 
title("Original Image (left), Noisy Image (middle), Output of Adaptive IIR Wiener Filter (right)");

% figure("Name", "Adaptive Wiener Filter Output");
% subplot(1,2,1); imshow(noisyRGBx); title("Noisy Initial image");
% subplot(1,2,2); imshow(RGBAdaptiveOut); 
% title("Output of Adaptive IIR Wiener Filter");

%--------------------------------------------------------------------------
% q3
%--------------------------------------------------------------------------

PSNRadaptive = psnr(RGBAdaptiveOut, x);
PSNRIIRWiener = psnr(RGBIIRWienerOut, x);

SSIMadaptive = ssim(RGBAdaptiveOut, x);
SSIMIIRWiener = ssim(RGBIIRWienerOut, x);

figure("Name", "Comparison-1");
subplot(1,2,1); imshow(x); title("Original Image");
subplot(1,2,2); imshow(noisyRGBx); title("Noisy Image");
figure("Name", "Comparison-2");
subplot(1,2,1); imshow(RGBIIRWienerOut); title("Output of IIR Wiener");
subplot(1,2,2); imshow(RGBAdaptiveOut); title("Output of Adaptive Wiener");

%--------------------------------------------------------------------------
% Functions used in this problem are below this line
%--------------------------------------------------------------------------
% Calculate the local mean
function locMean = localMean(image, n1, n2, M)
    locMean = mean(image(n1-M:n1+M, n2-M:n2+M), "all");
end

% Calculate the local variance
function locVar = localVar(image, n1, n2, M, locMean)
    locVar = mean((image(n1-M:n1+M, n2-M:n2+M) - locMean).^2, "all");
end

% Calculate noise variance for specified SNR value
function noiseVar = SNR2var(SNR, image)
    % Normalized image pixel values
    image = im2double(image);
    % Image pixel amount
    numPixels = size(image,1)*size(image,2);
    % Calculate the mean of the image
%     imageMean = mean(image, "all");
    % Calculate the variance of the image
%     imageVariance = (1/numPixels) * sum((image - imageMean).^2, "all");
    imageVariance = (1/numPixels) * sum((image).^2, "all");
    % Calculated noise variance using SNR formula inverted
    noiseVar = imageVariance / (10^(SNR/10));
end

% 2D-DFT for right alignment
function result = twodFFT(signal, nfft)
    % Apply padding at all sides to increase size up to DFT size
    padded = zeros(nfft, nfft);
    size1 = floor(size(signal,1)/2);
    % Deciding where to place the image or filter
    if mod(size(signal, 1),2)==0
        filterspace = nfft/2-(size1-1):nfft/2+size1;
    else
        filterspace = nfft/2-(size1-1):nfft/2+size1+1;
    end
    
    padded(filterspace,filterspace) = signal;
    % ifftshift to get the middle point up to (1,1) point
    padded = ifftshift(padded);
    
    result = fft2(padded, nfft, nfft);
end

% 2D-IDFT for right alignment
function result = twodIFFT(signalFFT, nfft, originalSize)
    % Take ifft and fix image by taking fftshift (we have taken ifftshift
    % at DFT calculation)
    signal = ifft2(signalFFT, nfft, nfft);
    signal = fftshift(signal);
    % Deciding which part of the image is the original image (to crop the
    % padding)
    size1 = floor(originalSize/2);
    if mod(originalSize,2)==0
        filterspace = nfft/2-(size1-1):nfft/2+size1;
    else
        filterspace = nfft/2-(originalSize-1):nfft/2+originalSize+1;
    end
    % Cropping the padding
    result = signal(filterspace, filterspace);
end

% 2D-DFT Magnitude and Phase Plot (used along the exercise)
function result = fft_plot(signal, plot_title, nfft)
    % Take FFT and shift using fftshift to center the DC component
    freq = twodFFT(signal, nfft);
    shifted_freq = fftshift(freq);
    % Calculate magnitude and phase
    magnitude = abs(shifted_freq);
    phase = angle(shifted_freq);
    
    % Find the end frequency to fix the axes
    ends = nfft/2;
    w1 = -ends:ends-1;
    w1 = w1./ends;
    w2 = w1;
    
    %Plot Magnitude
    plot_title = append(plot_title, ' ', "Magnitude & Phase");
    figure("Name", plot_title);
    % plotted wrt. DFT domain, just giving the starting k value.
    subplot(1,2,1);
    imagesc(w1, w2, log(1+magnitude));
    colormap(gray);
    title("Magnitude"); xlabel("\omega_1 (\times\pi rad/pixel)"); 
    ylabel("\omega_2 (\times\pi rad/pixel)");
    colorbar;
    set(gca,'dataAspectRatio',[1 1 1])
    axis on;
    
    %Plot Phase
    subplot(1,2,2);
    imagesc(w1, w2, phase);
    colormap(gray);
    title("Phase"); xlabel("\omega_1 (\times\pi rad/pixel)"); 
    ylabel("\omega_2 (\times\pi rad/pixel)");
    truesize([300 300]); colorbar;
    set(gca,'dataAspectRatio',[1 1 1])
    axis on;
    
    result = freq;
end