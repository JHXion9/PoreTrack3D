function [frames,descriptors,hrs] = func_psift(gss,oframes_pr,ohrs_pr)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
mmm=2;
magnif = 3.0 ;
NBP=8;
NBO=8;
verb=1;
frames      = [] ;
hrs=[];
descriptors = [] ;


    for o=1:gss.O
        oframes=oframes_pr{o};
        ohrs=ohrs_pr{o};
        % Compute the oritentations
        %     oframes = siftormx(...
        %         oframes, ...
        %         gss.octave{o}, ...
        %         gss.S, ...
        %         gss.smin, ...
        %         gss.sigma0 ) ;
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     oframes(4,:)=0;
        oframes(4,:)=0;
        oframes(3,:)=oframes(3,:);
        % Store frames
        x     = 2^(o-1+gss.omin) * oframes(1,:) ;
        y     = 2^(o-1+gss.omin) * oframes(2,:) ;
        sigma = 2^(o-1+gss.omin) * gss.sigma0 * 2.^(oframes(3,:)/gss.S) ;

        %     sigma=ones(size((oframes(3,:))));
        frames = [frames, [x(:)' ; y(:)' ; sigma(:)' ; oframes(4,:)] ];
        hrs=[hrs, ohrs(1,:)];
        
        
        % Descriptors
        % if nargout > 1
        %     if verb > 0
        %         fprintf('PSIFT: computing descriptors at oct %d...',o-1) ;
        %         tic ;
        %     end
        %     %             temp=(dogss.octave{o}-min(dogss.octave{o}(:)))*gss.S*2;
        %     temp=gss.octave{o};
        %     sh = siftdescriptor(...
        %         temp, ...
        %         oframes, ...
        %         gss.sigma0, ...
        %         gss.S, ...
        %         gss.smin, ...
        %         'Magnif',mmm* magnif, ...
        %         'NumSpatialBins', NBP, ...
        %         'NumOrientBins', NBO) ;
            
            
            
        %     descriptors = [descriptors, sh] ;
            
        %     if verb > 0, fprintf('done (%.3f s)\n',toc) ; end
        % end
    end



end

