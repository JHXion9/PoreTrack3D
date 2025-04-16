function face=func_maskPadding(src,mask,w,h)

src=imresize(src,[h,w]);
mask=imresize(mask,[h,w]);
se1=strel('diamond',80);%这里是创建一个结构元素
 mask=imerode(mask,se1);
se2=strel('diamond',20);%这里是创建一个对角线为10的菱形结构元素
mask=imdilate(mask,se2);

for i=1:h
    for j=1:w
        if mask(i,j)==255
            mask(i,j)=1;
        end
    end
end

im_crop=src.*mask;
for i=1:h
    for j=1:w
        if im_crop(i,j)==255
            im_crop(i,j)=0;
        end
    end
end
% b=edge(im_crop);
% [u,v]=find(b);
% edge1=[u,v];
face=im_crop;
figure(100)
imshow(face);
end