
addpath('bm3d_matlab_package/bm3d');

close all; clear all; clc
nfft = 1024;
% Read image
x = imread("F16.tiff");
xYcbcr = rgb2ycbcr(x);
y = double(xYcbcr(:, :, 1));

figure("Name", "Original image");
imshow(x); title("Original image");

% fft_plot(y, "Y Channel Image", nfft);

noiseVar = SNR2var(10, y);
noise = sqrt(noiseVar) * randn(size(y));
% noisedY = imnoise(y, "gaussian", 0, noiseVar/(255^2));
noisedY = noise + y; 

% snrVal = 10 * log10(var(y(:))/var(noise(:)));

% origPsnr = psnr(noisedY, y);
snrVal = snr(y(:), noise(:));

noisedRGB = xYcbcr;
noisedRGB(:, :, 1) = noisedY;
noisedRGB = ycbcr2rgb(noisedRGB);

figure("Name", "Noised image"); 
subplot(1,2,1); imshow(uint8(noisedY));
title("Noised Y Channel of the Image");
subplot(1,2,2); imshow(uint8(noisedRGB));
title("Noised RGB Image");


% Choosing [20, 424] to [94, 498] area to calculate sample variance 
% of noise. (for Bilateral filter)
flatPatch = noisedY(424:498, 20:94);

% Estimated noise variance calculated
% estNoiseVar = mean((flatPatch - mean(flatPatch, "all")).^2, "all");
% Degree of smoothing for bilateral filter
% DoS = 2 * estNoiseVar;

% Using parameters I have found
gaussDev = 1.7;
medianNeighbor = 7;
[peronaGradThreshold, peronaNumIter] = imdiffuseest(uint8(noisedY));
bilateralDegSmooth = 62700;
bilateralDev = 1.85;
nonLocDegSmooth = 82;
nonLocSearchWindow = 9;
nonLocCompWindow = 3;
BM3DDev = 0.2;
   
% Applying filters to the noisy Y channel
noisedY = uint8(noisedY);
YgaussianFiltered = imgaussfilt(noisedY, gaussDev);
YmedianFiltered = medfilt2(noisedY, [medianNeighbor medianNeighbor]);
YperonaFiltered = imdiffusefilt(noisedY,...
    "GradientThreshold", peronaGradThreshold,...
    "NumberOfIterations", peronaNumIter);
YbilateralFiltered = imbilatfilt(noisedY, bilateralDegSmooth,...
    bilateralDev);
YnonLocFiltered = imnlmfilt(noisedY,...
    "DegreeOfSmoothing", nonLocDegSmooth,...
    "SearchWindowSize", nonLocSearchWindow,...
    "ComparisonWindowSize", nonLocCompWindow);
YBM3DFiltered = uint8(255*BM3D(im2double(noisedY), BM3DDev));

YfilteredArray1 = [YgaussianFiltered YmedianFiltered];
YfilteredArray2 = [YperonaFiltered YbilateralFiltered];
YfilteredArray3 = [YnonLocFiltered YBM3DFiltered];

figure("Name", "Filtered Y Channel 1");
montage(YfilteredArray1);
title("Gaussian Filtered (left), Median Filtered (right)");

figure("Name", "Filtered Y Channel 2");
montage(YfilteredArray2);
title("Perona-Malik Filtered (left), Bilateral Filtered (right)");

figure("Name", "Filtered Y Channel 3");
montage(YfilteredArray3);
title("Non-local Means Filtered (left), BM3D Filtered (right)");


% PSNR & SSIM Calculation

PSNRGauss = psnr(YgaussianFiltered, uint8(y));
PSNRMedian = psnr(YmedianFiltered, uint8(y));
PSNRPerona = psnr(YperonaFiltered, uint8(y));
PSNRBilateral = psnr(YbilateralFiltered, uint8(y));
PSNRNonLoc = psnr(YnonLocFiltered, uint8(y));
PSNRBM3D = psnr(YBM3DFiltered, uint8(y));

SSIMGauss = ssim(YgaussianFiltered, uint8(y));
SSIMMedian = ssim(YmedianFiltered, uint8(y));
SSIMPerona = ssim(YperonaFiltered, uint8(y));
SSIMBilateral = ssim(YbilateralFiltered, uint8(y));
SSIMNonLoc = ssim(YnonLocFiltered, uint8(y));
SSIMBM3D = ssim(YBM3DFiltered, uint8(y));

% To detect the optimal parameters in an exhaustive way with for loops
% uintNoisy = uint8(noisedY);
% [PSNRGauss, paramGauss] = detectMaxPSNR("gauss", uintNoisy, uint8(y));
% [PSNRMedian, paramMedian] = detectMaxPSNR("median", uintNoisy, uint8(y));
% [PSNRPerona, paramPerona] = detectMaxPSNR("perona", uintNoisy, uint8(y));
% [PSNRBilateral, paramBilateral] = detectMaxPSNR("bilateral", uintNoisy, uint8(y));
% [PSNRNonLoc, paramNonLoc] = detectMaxPSNR("nlm", uintNoisy, uint8(y));
% [PSNRBM3D, paramBM3D] = detectMaxPSNR("BM3D", uintNoisy, uint8(y));


% Resulting RGB ImagesRGBgaussian = xYcbcr;
RGBgaussian = xYcbcr;
RGBgaussian(:, :, 1) = YgaussianFiltered;
RGBgaussian = ycbcr2rgb(RGBgaussian);

RGBmedian = xYcbcr;
RGBmedian(:, :, 1) = YmedianFiltered;
RGBmedian = ycbcr2rgb(RGBmedian);

RGBperona = xYcbcr;
RGBperona(:, :, 1) = YperonaFiltered;
RGBperona = ycbcr2rgb(RGBperona);

RGBbilateral = xYcbcr;
RGBbilateral(:, :, 1) = YbilateralFiltered;
RGBbilateral = ycbcr2rgb(RGBbilateral);

RGBnonLoc = xYcbcr;
RGBnonLoc(:, :, 1) = YnonLocFiltered;
RGBnonLoc = ycbcr2rgb(RGBnonLoc);

RGBBM3D = xYcbcr;
RGBBM3D(:, :, 1) = YBM3DFiltered;
RGBBM3D = ycbcr2rgb(RGBBM3D);

RGBfilteredArray1 = {RGBgaussian, RGBmedian};
RGBfilteredArray2 = {RGBperona, RGBbilateral};
RGBfilteredArray3 = {RGBnonLoc, RGBBM3D};

figure("Name", "Filtered RGB Images 1");
montage(RGBfilteredArray1, "Size", [1 2]);
title("RGB Images Gaussian Filtered (left), Median Filtered (middle)");

figure("Name", "Filtered RGB Images 2");
montage(RGBfilteredArray2, "Size", [1 2]);
title("RGB Images Perona-Malik Filtered (left), Bilateral Filtered (right)");

figure("Name", "Filtered RGB Images 3");
montage(RGBfilteredArray3, "Size", [1 2]);
title("Non-local Means Filtered (middle), BM3D Filtered (right)");


% Find maximum PSNR giving deviation parameter
function [maxPSNR, maxParams] = detectMaxPSNR(func, noisy, original)
    maxPSNR = 0;
    % Find Gauss parameters, found once, too much computation time
    if strcmp(func, "gauss")
        for dev=0.6:0.01:3
            YgaussianFiltered = uint8(imgaussfilt(noisy, dev));
            PSNRGauss = psnr(YgaussianFiltered, uint8(original));
            if PSNRGauss > maxPSNR
                maxPSNR = PSNRGauss;
                maxParams = {"stdDev", dev};
            end
        end
    % Find Median parameters, found once, too much computation time
    elseif strcmp(func, "median")
        for neighbor=1:20
            YmedianFiltered = uint8(medfilt2(noisy, [neighbor neighbor]));
            PSNRMedian = psnr(YmedianFiltered, uint8(original));
            if PSNRMedian > maxPSNR
                maxPSNR = PSNRMedian;
                maxParams = {"neighbour", neighbor};
            end
        end
    % Find Perona parameters, found once, too much computation time
    elseif strcmp(func, "perona")
        [gradThreshold, numIter] = imdiffuseest(noisy);
        YperonaFiltered = uint8(imdiffusefilt(noisy,...
            "GradientThreshold", gradThreshold,...
            "NumberOfIterations", numIter));
        maxPSNR = psnr(YperonaFiltered, uint8(original));
        maxParams = {"gradThreshold", gradThreshold; "numIter", numIter};
    % Find Bilateral parameters, found once, too much computation time    
    elseif strcmp(func, "bilateral")
        for degSmooth=62500:10:63500
            for sigma=1:0.01:2
                YbilateralFiltered = uint8(imbilatfilt(noisy, degSmooth, sigma));
                PSNRBilateral = psnr(YbilateralFiltered, uint8(original));
                if PSNRBilateral > maxPSNR
                    maxPSNR = PSNRBilateral;
                    maxParams = {"degSmooth", degSmooth; "stdDev", sigma};
                end
            end
        end
    % Find Non-local parameters, found once, too much computation time    
    elseif strcmp(func, "nlm")
        for degSmooth=80:1:140
            for searchWindow=1:2:11
                for compWindow=1:2:searchWindow
                    YnonLocFiltered = uint8(imnlmfilt(noisy,...
                        "DegreeOfSmoothing", degSmooth,...
                        "SearchWindowSize", searchWindow,...
                        "ComparisonWindowSize", compWindow));
                    PSNRnonLoc = psnr(YnonLocFiltered, uint8(original));
                    if PSNRnonLoc > maxPSNR
                        maxPSNR = PSNRnonLoc;
                        maxParams = {"degSmooth", degSmooth;...
                                     "searchWindowSize", searchWindow;...
                                     "comparisonWindowsSize", compWindow};
                    end
                end
            end
        end
    % Find BM3D parameters, found once, too much computation time    
    elseif strcmp(func, "BM3D")
        noisy = im2double(noisy);
        for sigma=0.11:0.01:0.25
            YBM3DFiltered = uint8(255*BM3D(noisy, sigma));
            PSNRBM3D = psnr(YBM3DFiltered, uint8(original));
            if PSNRBM3D > maxPSNR
                maxPSNR = PSNRBM3D;
                maxParams = {"stdDev", sigma};
            end
        end
    end
end

% Calculate noise variance for specified SNR value
function noiseVar = SNR2var(SNR, image)
    % Image pixel amount
    numPixels = size(image,1)*size(image,2);
    % Calculate the mean of the image
%     imageMean = mean(image, "all");
    % Calculate the variance of the image
%     imageVariance = (1/(numPixels-1)) * sum((image - imageMean).^2, "all");
    imageVariance = (1/(numPixels-1)) * sum((image).^2, "all");

    % Calculated noise variance using SNR formula inverted
    noiseVar = imageVariance / (10^(SNR/10));
end
