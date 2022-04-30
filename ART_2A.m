AreaMatrix = zeros(length(SAMPLE),500);
for i = 1:length(SAMPLE)
    idx = SAMPLE(i).pkl(abs(SAMPLE(i).pkl(:,1)) <= 250,1:2);
    idx(idx == 0,:) = [];
    for j = 1:size(idx,1)
        if idx(j,1) < 0
            idx(j,1) =  -idx(j,1);
        elseif idx(j,1) > 0
            idx(j,1) = idx(j,1) + 250;
        end
    end
    for k = 2:size(idx,1)
        idx(k,2) = idx(k,2) + double(idx(k,1) == idx(k - 1,1)) * idx(k - 1,2);
    end
    AreaMatrix(i,idx(:,1)) = idx(:,2);
end
clear i j k idx

[MCell, MCount, OWM, FraC] = re_art2a_S(AreaMatrix, 2, 0.05, 0.8);