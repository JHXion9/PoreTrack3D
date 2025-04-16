function func_plot_scale_color_grayImg( I, frames, scale_bin, dogss, ...
    pr_idx, outputpath,rank)
%FUNC_PLOT_SCALE_COLOR Summary of this function goes here
% scale_bin is the predefined color bin used in
%   Detailed explanation goes here
% I shows clr jet colormap pts by scale_bin
% dogss shows the corresponding sig pts (red)
% close all;
% figure
% g=imshow(imresize( sigma_thresh_hist./max(sigma_thresh_hist(:)),10,'nearest'),[]);
% print(gcf, '-dpng',  [outputpath 'sigma_thresh_hist.png']);

Noscale_bin=length(scale_bin);
[imsz1 imsz2]=size(I);
omin = dogss.omin ;
smin = dogss.smin ;
nlevels = dogss.smax-dogss.smin+1 ;
scale_bin_f=[];


for i=1:dogss.O %4
    for j=0:dogss.S-1% 5
        scale_bin_f=[scale_bin_f 2^(i-1+omin) * dogss.sigma0*2.^(j/dogss.S) ];
    end
end
k=pr_idx; %1:Nothresh_bin

for i=1:Noscale_bin-1
    scale_bin(i)=(scale_bin(i)+scale_bin(i+1))/2;
end
tempNo=0;
close
figure(1);
colormap(gray);
clf;
tightsubplot(1, 1) ;
imagesc(1:imsz2,1:imsz1,I) ; axis image ; axis off ;
for oi=1:dogss.O
    for si=1:nlevels
        Nopts=0;
        if si~=1 && si~=nlevels
            tempNo=tempNo+1;
            hold on
            for idx=1:Noscale_bin
                if idx==1
                    sel=frames(3,:)<scale_bin(idx) ;
                else if idx==Noscale_bin
                        sel=frames(3,:)>=scale_bin(idx-1);
                    else
                        sel=frames(3,:)>=scale_bin(idx-1)&frames(3,:)<scale_bin(idx);
                    end
                end
                if tempNo~=1
                    sel=sel&(frames(3,:)>=scale_bin_f(tempNo-1)&frames(3,:)<scale_bin_f(tempNo));
                else
                    sel=sel&(frames(3,:)<scale_bin_f(tempNo));
                end
                if sum(sel)~=0
%                     disp(num2str(sum(sel)));
                    Nopts=Nopts+sum(sel);
                end
                clr=jet(Noscale_bin);% help colormap
                
%                 figure(idx);
                
%                  plot(x,y);
%                   figure(1);
                  figure(Noscale_bin+1);
                h=plot(frames(1,sel),frames(2,sel),'.', 'MarkerSize',min([max([3 round(imsz2/80)]) 6]),'Color', clr(idx,:)) ;
%                 pointx = get(h,'XData')
%                 pointy = get(h,'YData')
%                 point_outputpath=sprintf('point_%d.txt',idx);
%                 fid=fopen(point_outputpath,'w');
%                 fprintf(fid,'%d %d\n',pointx,pointy);
%                 fclose(fid);
            end
            hold off
            
        end
        
    end
end
if imsz1>1200 && imsz2>1200
    imsz1=round(imsz1/2);
    imsz2=round(imsz2/2);
end
if imsz1<400 && imsz2<400
    imsz1=round(imsz1*1.5);
    imsz2=round(imsz2*1.5);
end

% F=getframe(gcf);
% imwrite(F.cdata , outputpath,'png');
% figure(Noscale_bin+1);
% % figure_path=sprintf('./result/point_%d.fig',rank)
% % output_path=sprintf('./result/point_%d.txt',rank)
% figure_path=sprintf('./15_1point.fig');
% % output_path=sprintf('./15_1point.txt');
% % savefig(figure_path);
% % 
% % px=frames(1,:);
% % py=frames(2,:);
% % px=round(px);
% % py=round(py);
% % point=[px;py]
% % [m,n]=size(point)
% % fid=fopen(output_path,'w');
% % for i=1:n
% % fprintf(fid,'%i  ',point(1,i));
% % fprintf(fid,'%i\n',point(2,i));
% % end
% % fclose(fid);
% write_point(figure_path,output_path);

% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperUnits', 'points');
% set(gcf, 'PaperPosition', [0 0 imsz2/2 imsz1/2]);
% print(gcf,'-dpng',outputpath);%,'-r525'








% [temp, ind]=find(oframes(3,:)<centroid/kkk,1,'last');
% if isempty(ind)
%     ind=2;
% end
% [temp, ind2]=find(oframes(3,:)>centroid*kkk,1);
% if isempty(ind2)
%     ind2=length((oframes(3,:)))-1;
% end
% colorseg=[ind ind2];
% %     showImage(borderWidth, imresize(I,0.5), flipud(oframes(1:2,:)).*0.5,colorseg, [num2str(k) '.png']);
% writeImage(borderWidth, I,oframes,colorseg, [outputpath num2str(k) '.png']);



% end



