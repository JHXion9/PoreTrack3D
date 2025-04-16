function [frames,oframes_pr,ohrs_pr] = func_detect_dogss_pr(dogss,gss,pr)

% detect and descriptor were seperated to 2 function, thus oframes_pr has to
% be set 3x1 octaves size cell
verb=1;
% pore_ratio_list=0.0003:0.0003:0.09;
S=dogss.S;
omin   = dogss.omin;
% O      = floor(log2(min(M,N)))-omin-6 ; % up to 512x512 images
O = dogss.O;
dist_eyes=1/1.6;
sigma0 = dogss.sigma0;                  % smooth lev. -1 at 1.6
kk=2^(1/S) ;

sigman=0.5;

thresh = (2^(1/S)-1)/(2^(1/S)+1) ;% BianT
r      = 3 ;
% r=22.9564392373896;
NBP    = 8 ;%used to remove boundary points
% NBO    = 8 ;
magnif = 3.0 ;
discard_boundary_points = 1 ;

frames= [];

sigma_mean = 0.0;

if verb > 0
    fprintf('PSIFT detector parameters\n')
    fprintf('  thersh [Threshold]     : %e\n', thresh) ;
    fprintf('       r [EdgeThreshold] : %.3f\n', r) ;
    fprintf('SIFT descriptor parameters\n')
    fprintf('  magnif [Magnif]        : %.3f\n', magnif) ;
    fprintf('     NBP [NumSpatialBins]: %d\n', NBP) ;
    %fprintf('     NBO [NumOrientBins] : %d\n', NBO) ;
end

for o=1:gss.O
    if verb > 0
        fprintf('PSIFT: processing octave %d\n', o-1+omin) ;
        tic ;
    end
    
    % Local maxima of the DOG octave
    % The 80% tricks discards early very weak points before refinement.
    
    
    
    
    idx= siftlocalmax(  dogss.octave{o}, 0.8*pr*thresh  ) ;% dark point
    %   idx = [idx , siftlocalmax( - dogss.octave{o}, thresh)] ; % light point
    K=length(idx) ;
    [i,j,s] = ind2sub( size( dogss.octave{o} ), idx ) ;
    y=i-1 ;
    x=j-1 ;
    s=s-1+dogss.smin ;
    oframes = [x(:)';y(:)';s(:)'] ;
    
    if verb > 0
        fprintf('PSIFT: %d initial points (%.3f s)\n', ...
            size( oframes, 2), toc) ;
        tic ;
    end
    
    % Remove points too close to the boundary
    if discard_boundary_points
        % radius = maginf * sigma * NBP / 2
        % sigma = sigma0 * 2^s/S
        
        rad = magnif * gss.sigma0 * 2.^(oframes(3,:)/gss.S) * NBP / 2 ;
        sel=find(...
            oframes(1,:)-rad >= 1                     & ...
            oframes(1,:)+rad <= size(gss.octave{o},2) & ...
            oframes(2,:)-rad >= 1                     & ...
            oframes(2,:)+rad <= size(gss.octave{o},1)     ) ;
        oframes=oframes(:,sel) ;
        
        if verb > 0
            fprintf('PSIFT: %d away from boundary\n', size(oframes,2)) ;
            tic ;
        end
    end
    
    % Refine the location, threshold strength and remove points on edges
    [oframes_pr{o},ohrs_pr{o}] = siftrefinemx_lee(...
        oframes, ...
        dogss.octave{o}, ...
        dogss.smin, ...
        pr*thresh, ...%pr*thresh
        r) ;% after this the oframes ori is not integer
    
    
    if verb > 0
        fprintf('PSIFT: after removing points on edges, %d refined at oct %d, (%.3f s)\n', ...
            size(oframes_pr{o},2), o-1+omin,toc) ;
        tic ;
    end
    % Without Compute the oritentations
    % Store frames
    x     = 2^(o-1+gss.omin) * oframes_pr{o}(1,:) ;
    y     = 2^(o-1+gss.omin) * oframes_pr{o}(2,:) ;
    sigma = 2^(o-1+gss.omin) * gss.sigma0 * 2.^(oframes_pr{o}(3,:)/gss.S);
    
    %%%
%     sigma_mean = mean(sigma);
%     x = x(sigma > sigma_mean * 0.95);
%     y = y(sigma > sigma_mean * 0.95);
%     sigma = sigma(sigma > sigma_mean * 0.95);
    %%%
  
    temp=zeros(size(x(:)'));
    frames = [frames, [x(:)' ; y(:)' ; sigma(:)' ; temp] ] ;
    
end

if verb > 0
    fprintf('PSIFT: total %d keypoints have been detected,\n', ...
        size(frames,2)) ;
    
end


end

