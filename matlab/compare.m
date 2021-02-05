ref_img = '../checkpoints/companding_v1/datasets/a/1.png';
A_img = '../checkpoints/companding_v1/results/companding_GANs_16_expanded_1.png';
B_img = '../checkpoints/companding_v1/results/companding_GANs_16_compressed_1.png';

ref = imread(ref_img);
A = imread(A_img);
B = imread(B_img);

figure
montage({ref_img, B_img, A_img})
title('Reference Image (Left) vs. Compressed Image (Center) vs. Expanded Image (Right)')

[ssimval,ssimmap] = ssim(A, ref);
[ssimval_comp,ssimmap_comp] = ssim(B, ref);

figure
imshow(ssimmap,[])
title(['(Expanded) Local SSIM Map with Global SSIM Value: ', num2str(ssimval)])

figure
imshow(ssimmap_comp,[])
title(['(Compressed) Local SSIM Map with Global SSIM Value: ', num2str(ssimval_comp)])