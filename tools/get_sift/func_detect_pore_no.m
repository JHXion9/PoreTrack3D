function [frames,descriptors,pore_ratio_list,hrs] = func_detect_pore_no(I,no_kpttarget,outputno)
% date:2023.10.31 name:Li Dong
% pr list 1.random select pr 2. pre gradient select 2nd pr 3. gradient
% select 3rd 4.check if error 10% kpt no. if not choose 4th pr

% 0.05 9000
% 0.055 7000
%pore_ratio_list is nx5 table, 1st  upperbound=0.09  2nd lastkpt=-1 3rd up=1/low=3
%lowerbound=0.001 kpt=-1
pore_ratio_list=[0.02 -1 0.00001 100000 -1];% pr biggest, smallest kptno, pr smallest, biggest kptno

verb=1;
saveeverypr=0;


% 差分的图像，和原本的高斯金字塔
[dogss,gss]=func_dog(I);

outputpath=sprintf('Afunc_detect_pore_no%d.png',outputno)
kpterror=1;
pr_idx=0;
no_new=-1;
no_old=-2;
while abs(kpterror)>0.1&&(no_new~=no_old)
    pr_idx=pr_idx+1;
    if pr_idx==1%pr biggest
        % 1st select pr =0.05
        no_old=pore_ratio_list(2);
        pr_new=pore_ratio_list(1);
    else
        no_old=no_new;
    end
    [frames,oframes_pr, ohrs_pr]=func_detect_dogss_pr(dogss,gss,pr_new);
    no_new=size(frames,2);
    if no_new>pore_ratio_list(2)&&no_new<pore_ratio_list(4) % if kptno in the boundary
        pr_old=pr_new;
        if no_new<no_kpttarget % if no_new still smaller than target
            pore_ratio_list(1:2)=[pr_new no_new];% update the biggest pr 
            pore_ratio_list(5)=1;%up boundry
        else
            pore_ratio_list(3:4)=[pr_new no_new];%update the smallest pr
            pore_ratio_list(5)=3;%low boundry
        end
    else
        fprintf('over up/low bound, do nothing.\n');
    end
    kpterror=(no_new-no_kpttarget)/no_kpttarget;
    
    if abs(kpterror)<0.3
        pr_new=pr_old+0.01*kpterror;%pr_idx bigger, update add/minus smaller
        if pr_new<=pore_ratio_list(3)
            pr_new=pore_ratio_list(3);
        end
            
    else
        pr_new=(pore_ratio_list(1)+pore_ratio_list(3))/2;% 
        
    end
    
    
    if saveeverypr>0
        oframes_cell{pr_idx}=oframes_pr;
    end
    
    
end
%pr_new

scale_bin= [0.5*0.4*1.6*2^0.2:0.1:1 1.1:0.2:2.1 2.5:0.5:4.5 5:1:10];%4*1*1.6*2
func_plot_scale_color_grayImg( I, frames, scale_bin, dogss, ...
    pr_idx, outputpath,outputno)
[frames,descriptors,hrs] = func_psift(gss,oframes_pr,ohrs_pr);


end

