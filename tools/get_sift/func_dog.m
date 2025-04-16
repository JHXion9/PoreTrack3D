function [dogss,gss] = func_dog(I)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
verb=1;
sigman=0.5;
O=3;
S=8;%No. of layers
omin=0;
kk=2^(1/S) ;
sigma0 = 1/1.6*1.6*2^(1/S);                  % smooth lev. -1 at 1.6

if verb>0, fprintf('PSIFT: computing scale space...') ; tic ; end
img=double(I)./255.0;
gss = gaussianss_lee(img,sigman,O,S,omin,-1,S+1,sigma0,kk) ;

if verb>0, fprintf('(%.3f s gss; ',toc) ; tic ; end

dogss = diffss(gss) ;

if verb > 0
    fprintf('PSIFT scale space parameters [PropertyName in brackets]\n');
    fprintf('  sigman [SigmaN]        : %f\n', sigman) ;
    fprintf('  sigma0 [Sigma0]        : %f\n', dogss.sigma0) ;
    fprintf('       O [NumOctaves]    : %d\n', dogss.O) ;
    fprintf('       S [NumLevels]     : %d\n', dogss.S) ;
    fprintf('    omin [FirstOctave]   : %d\n', dogss.omin) ;
    fprintf('    smin                 : %d\n', dogss.smin) ;
    fprintf('    smax                 : %d\n', dogss.smax) ;
end
end

