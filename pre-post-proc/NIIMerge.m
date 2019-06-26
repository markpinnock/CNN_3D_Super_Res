HI_PATH = "F:\PhD\Super_Res_Data\Toshiba_Vols\NII_Test\Hi\";
LO_PATH = "F:\PhD\Super_Res_Data\Toshiba_Vols\NII_Test\Lo\";
SAVE_PATH = "F:\PhD\Super_Res_Data\Toshiba_Vols\NII_Test\";

hi_list = dir(HI_PATH);
hi_list = hi_list(3:end);
lo_list = dir(LO_PATH);
lo_list = lo_list(3:end);

subject = 'UCLH_21093614';
up_to_vol = 9;
rest_vol = 20;

hi_out_1 = zeros(512, 512, 12 * up_to_vol);
lo_out_1 = zeros(512, 512, 12 * up_to_vol);
int_out_1 = zeros(512, 512, 12 * up_to_vol);
hi_out_2 = zeros(512, 512, 12 * rest_vol);
lo_out_2 = zeros(512, 512, 12 * rest_vol);
int_out_2 = zeros(512, 512, 12 * up_to_vol);
[X Y Z] = ndgrid(1:512, 1:512, 2:4:12);
[Xq Yq Zq] = ndgrid(1:512, 1:512, 1:12);

count = 0;
save_idx = 1;

for i = 1:length(hi_list)
    
    if ~contains(hi_list(i).name, subject)
        continue;
    end
    
    if count + 1 > up_to_vol
        break;
    end
    
    hi_vol = niftiread(strcat(HI_PATH, hi_list(i).name));
    lo_vol = niftiread(strcat(LO_PATH, lo_list(i).name));

    dnsmp_lo_vol = lo_vol(:, :, 2:4:12);
    dnsmp_lo_vol = imgaussfilt3(dnsmp_lo_vol, [1 1 1]);

    int_vol = interpn(X, Y, Z, dnsmp_lo_vol, Xq, Yq, Zq, 'spline');
    
    hi_out_1(:, :, (save_idx * 12 - 11):(save_idx * 12)) = hi_vol;
    lo_out_1(:, :, (save_idx * 12 - 11):(save_idx * 12)) = lo_vol;
    int_out_1(:, :, (save_idx * 12 - 11):(save_idx * 12)) = int_vol;

    fprintf("%s %s\n", hi_list(i).name, lo_list(i).name);
    save_idx = save_idx + 1;
    count = count + 1;
end

niftiwrite(hi_out_1, strcat(SAVE_PATH, subject, "_1_1", '_H.nii'));
niftiwrite(lo_out_1, strcat(SAVE_PATH, subject, "_1_1", '_L.nii'));
niftiwrite(int_out_1, strcat(SAVE_PATH, subject, "_1_1", '_I.nii'));
fprintf("SAVED\n");
save_idx = 1;

for i = i:length(hi_list)
    
    if ~contains(hi_list(i).name, subject)
        break;
    end

    hi_vol = niftiread(strcat(HI_PATH, hi_list(i).name));
    lo_vol = niftiread(strcat(LO_PATH, lo_list(i).name));

    dnsmp_lo_vol = lo_vol(:, :, 2:4:12);
    dnsmp_lo_vol = imgaussfilt3(dnsmp_lo_vol, [1 1 1]);

    int_vol = interpn(X, Y, Z, dnsmp_lo_vol, Xq, Yq, Zq, 'spline');
    
    hi_out_2(:, :, (save_idx * 12 - 11):(save_idx * 12)) = hi_vol;
    lo_out_2(:, :, (save_idx * 12 - 11):(save_idx * 12)) = lo_vol;
    int_out_2(:, :, (save_idx * 12 - 11):(save_idx * 12)) = int_vol;

    fprintf("%s %s\n", hi_list(i).name, lo_list(i).name);
    save_idx = save_idx + 1;
end

niftiwrite(hi_out_2, strcat(SAVE_PATH, subject, "_1_2", '_H.nii'));
niftiwrite(lo_out_2, strcat(SAVE_PATH, subject, "_1_2", '_I.nii'));
niftiwrite(int_out_2, strcat(SAVE_PATH, subject, "_1_2", '_L.nii'));
fprintf("SAVED\n");

