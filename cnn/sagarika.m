function cnn
%load('sag1vars.mat','w1','b1','w2','b2','fw1','bf1','fw2','bf2','fw3','bf3');
path=strcat(pwd,'/steering');
%path='F:\steering';
data='data.txt';
data_path =     fullfile(path,data)
fileID = fopen(data_path,'r');
c=textscan(fileID,'%s %f');

image=c{1};
angle=c{2};

addpath(path);
N=size(c{1});

epochs=2000;
eta=0.001;
prob=0.5;
no_of_kernels=5;
kernel_1=[];
%for i=1:no_of_kernels
%    kernel_1[1:5,1:5,1:3,i]=-0.3+(0.6)*rand(5,5,3);
%end
for epoch = 1:2000
err = 0;
for inst=1:21999
    name=(strsplit(image{inst},'./'));
   
    name=char(name(2));   
    ang = angle(inst);
    pixel=imread(name);
    
    %disp(pixel);
    %size(pixel)%32x32x3



%generating kernel

w1 = -0.3+(0.6)*rand(5,5,3,5);
b1 = -0.3+(0.6)*rand(1,5);

%b1size=size(b1)
%disp(w);
%disp(##)
for i=1:5
   cc1(:,:,i)=convn(pixel,w1(:,:,:,i),'valid')+b1(i);
end    
%cc1size=size(cc1)%28x28x5
%applyig relu

for i=1:28
    for j=1:28
        for z=1:5
            if(cc1(i,j,z)<0)
                cc1(i,j,z)=0;
            end
            
        end
    end
end
%disp(cc1);
%size(cc1)%28x28x5

%APPLYING MAX POLLING
for z=1:5
    for i=1:14
        for j=1:14
            %a(i,j,z)=k;
            b=cc1(2*i-1:2*i,2*j-1:2*j,z);
            p1(i,j,z)=max(max(b));
            %k=k+1;
        end
    end
end
%disp(p);
p1size=size(p1);%14x14x5


%initialising kernel2 
w2= -0.3+(0.6)*rand(5,5,5,10);
b2= -0.3+(0.6)*rand(1,10);
for i=1:10
   cc2(:,:,i)=convn(p1,w2(:,:,:,i),'valid')+b2(i);
end
cc2size=size(cc2);%10x10x10

%applyig relu on cc2

for i=1:10
    for j=1:10
        for z=1:10
            if(cc2(i,j,z)<0)
                cc2(i,j,z)=0;
            end
            
        end
    end
end
%disp(cc1);
%size(cc1)%28x28x5

%APPLYING MAX POLLING
for z=1:10
    for i=1:5
        for j=1:5
            %a(i,j,z)=k;
            b=cc2(2*i-1:2*i,2*j-1:2*j,z);
            p2(i,j,z)=max(max(b));
            %k=k+1;
        end
    end
end
%disp(p);
p2size=size(p2);%5x5x10
input=reshape(p2,250,1);
sizeinput=size(input);%250x1


%Fully connected 
fw1= -0.3+(0.6)*rand(128,250);
bf1= -0.3+(0.6)*rand(128,1);

fw2= -0.3+(0.6)*rand(64,128);
bf2= -0.3+(0.6)*rand(64,1); 

fw3= -0.3+(0.6)*rand(1,64);
bf3= -0.3+(0.6)*rand(1,1);


F1 = (fw1*input) + bf1;
F1 = 1./(1.+exp(F1));

F2 = (fw2*F1) + bf2;
F2 = 1./(1.+exp(F2));

F3 = (fw3*F2) + bf3;


%backpropagation

delF3 = F3 - ang;
delfw3 = delF3*F2';
delbf3 = delF3;

delF2 = fw3'*delbf3;
delbf2 = delF2.*F2.*(1-F2);
delfw2 = delbf2*F1';

delF1 = fw2'*delbf2;
delbf1 = delF1.*F1.*(1-F1);
delfw1 = delbf1*input';

delinput = fw1'*delF1;

delin = reshape(delinput,5,5,10);

for z=1:10
    for i=1:5
        for j=1:5
            
            b = zeros(2,2);
            b(find(cc2(2*i-1:2*i,2*j-1:2*j,z)==p2(i,j,z))) = delin(i,j,z); 
            delp2(2*i-1:2*i,2*j-1:2*j,z) = b;
            
        end
    end
end

delp2(find(cc2 == 0)) = 0;

delw2 = zeros(size(w2));

for q = 1:10
    delb2(q) = sum(sum(delp2(:,:,q)));
end

for q = 1:10
    delw2(:,:,:,q) = convn(p1,delp2(:,:,q),'valid');
end

delp1 = zeros(size(p1));

for q = 1:10
    delp1 = delp1 + convn(delp2(:,:,q),rot90(rot90(w2(:,:,:,q))),'full');
end

for z=1:5
    for i=1:14
        for j=1:14
            b = zeros(2,2);
            b(find(cc1(2*i-1:2*i,2*j-1:2*j,z)==p1(i,j,z))) = delp1(i,j,z); 
            delr1(2*i-1:2*i,2*j-1:2*j,z) = b;
        end
    end
end

delr1(find(cc1==0)) = 0;

for q = 1:5
    delw1(:,:,:,q) = convn(pixel,delr1(:,:,q),'valid');
end

for q = 1:5
    delb1(q) = sum(sum(delr1(:,:,q)));
end

w1 = w1 - (eta*delw1);
b1 = b1 - (eta*delb1);
w2 = w2 - (eta*delw2);
b2 = b2 - (eta*delb2);
fw1 = fw1 - (eta*delfw1);
bf1 = bf1 - (eta*delbf1);
fw2 = fw2 - (eta*delfw2);
bf2 = bf2 - (eta*delbf2);
fw3 = fw3 - (eta*delfw3);
bf3 = bf3 - (eta*delbf3);
err = err + (delF3)^2;

if(mod(inst,1000) == 0)
    fprintf(int2str(1));
end

end
epoch
err/21999
disp(['k1',find(w1 == Inf),find(isnan(w1))
        'k2',find(w2 == Inf)
        ,find(isnan(w2))
        'w1',find(fw1 == Inf)
        ,find(isnan(fw1))
        'w2',find(fw2 == Inf)
        ,find(isnan(fw2))])
save('sag1vars2.mat','w1','b1','w2','b2','fw1','bf1','fw2','bf2','fw3','bf3');
end