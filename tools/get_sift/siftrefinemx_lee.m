%   double* buffer = (double*) mxMalloc(K*3*sizeof(double)) ;
%     double* buffer2 = (double*) mxMalloc(K*1*sizeof(double)) ;
%     double* bufferD = (double*) mxMalloc(K*3*sizeof(double)) ;
%     double* buffer2D = (double*) mxMalloc(K*1*sizeof(double)) ;
%     
%     D means delete
%     2 means harris 
%     
%     double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;/*trace^2 / Det*/
%           double harris_score=(Dxx*Dyy - Dxy*Dxy)-0.04*(Dxx+Dyy)*(Dxx+Dyy);
%           
%           
%                 *buffer_iterator++ = xn ;
%             *buffer_iterator++ = yn ;
%             *buffer_iterator++ = sn+smin  ;
%             *buffer_iterator2++ = harris_score ;
%           