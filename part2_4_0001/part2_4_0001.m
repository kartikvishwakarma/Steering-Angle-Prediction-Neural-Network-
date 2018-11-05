function ann
%path='C:\Users\kartik\Documents\MATLAB\steering\';
clear all;
%path = 'F:\steering';
path='/home/clab2/workspace/steering';
data='data.txt';
data_path =     fullfile(path,data);
fileID = fopen(data_path,'r');

c=textscan(fileID,'%s %f');

image=c{1};
angle=c{2};
addpath(path);
N=size(c{1});

A=[];
for i=1:N
    A=[A; image(i) angle(i)];
end

rand_A=A(randperm(N(1)),:);
valid=[];
train=[];

for i=1:N*.2
    valid=[valid; rand_A(i,:)];
end

for i=(N*.2+1):N 
    train=[train; rand_A(i,:)];
end


w1=-0.3+(0.6)*rand(128,(1024+1));
w2=-0.3+(0.6)*rand(64,(128+1));
w3=-0.3+(0.6)*rand(1,(64+1));


disp('train function Called');
[W1, W2, W3, ERROR]= ann_train(train, valid, w1, w2, w3);
disp('train function Returned');


disp('validation function Called');
%test_error=ann_test(test, test_size, W1, W2, W3);
disp('validation function Returned');


function test_error = ann_test(test, test_size, W1, W2, W3)
 x=[];
 y=[];
    for i=1:test_size
        dataset=test(i,:);
        name = strsplit(dataset{1}, './');
        name = char(name(2));
        X=imread(name);
        X=rgb2gray(X);
        X=reshape(X,[1024,1]);
        X=double(X);
        X=X/max(X);
        x=[x X];
        Y=dataset{2};
        y=[y Y];
    end
        x1=1./(1.+exp(-W1*[ones(1,test_size); x]));
        x2=1./(1.+exp(-W2*[ones(1,test_size); x1]));
        x3=W3*[ones(1,test_size);x2];
        error=(x3-y).*(x3-y);
        
        disp('Error!!!');
        test_error = error/2;
        dlmwrite('test_error.txt', error/2, 'delimiter','\t');
return; 



function [ w1, w2, w3, error ] = ann_train(train, valid, w1, w2, w3)

epochs=1000;
eta=0.001;
batch_size=64;
train_size=size(train,1);
valid_size=size(valid,1);
E=0;
error=0;
x=[];
y=[];
v_x=[];
v_y=[];
file=fopen('part2_4_0001_error.txt','a');

for i=1:epochs
%--------------------validation_model-------------------------------%    
    v_error=[];
    for vk=1:batch_size:floor(valid_size/batch_size)*batch_size
        
        for b=vk:vk+batch_size-1
            dataset=valid(b,:);
            name=(strsplit(dataset{1},'./'));
            name=char(name(2));
            X=imread(name);
            %imshow(x);
            X=rgb2gray(X);
            X=reshape(X,[1024,1]);
            X=double(X);
            X=X/(max(X));
            v_x=[v_x X];    %1024x10
            Y=dataset{2};
            v_y=[v_y Y];    %1x10                 
        end
        
        v_x1=1./(1.+exp(-w1*[ones(1,batch_size); v_x]));
        
        
       
        
        v_x2=1./(1.+exp(-w2*[ones(1,batch_size); v_x1]));
        
        
        v_x3=w3*[ones(1,batch_size);v_x2];
        E=(v_x3-v_y).*(v_x3-v_y);
        v_error=[v_error sum(E)];
        v_x=[];
        v_y=[];
    end
%--------------------train_model------------------------------------%    
    error=[];
    for k=1:batch_size:floor(train_size/batch_size)*batch_size
        
        for b=k:k+batch_size-1
            dataset=train(b,:);
            name=(strsplit(dataset{1},'./'));
            name=char(name(2));
            X=imread(name);
            %imshow(x);
            X=rgb2gray(X);
            X=reshape(X,[1024,1]);
            X=double(X);
            X=X/(max(X));
            x=[x X];    %1024x10
            Y=dataset{2};
            y=[y Y];    %1x10                 
        end
        
       
    
        x1=1./(1.+exp(-w1*[ones(1,batch_size); x]));
        x2=1./(1.+exp(-w2*[ones(1,batch_size); x1]));
       
        x3=w3*[ones(1,batch_size);x2];
        
        delta2=(x3-y);
        dE3=delta2*[ones(1,batch_size);x2]';
        
        delta1 = (w3(:,2:end)'*delta2).*(x2.*(1-x2));
        dE2 = delta1*[ones(1,batch_size);x1]';
        
        delta0 = (w2(:,2:end)'*delta1).*(x1.*(1-x1));
        dE1 = delta0*[ones(1,batch_size);x]';

        w3=w3-eta*dE3;
        w2=w2-eta*dE2;
        w1=w1-eta*dE1;
    
        E=((x3-y).*(x3-y));
        
    
        x=[];
        y=[];
    
       
        
        error=[error; sum(E)];
       
    end
    sprintf('Epochs......%d',i)
    
    fprintf(file,'%f        %f      %d\n', sum(error)/(2*train_size),sum(v_error)/(2*valid_size),i);
   
end
w1=w1;
w2=w2;
w3=w3;
error=E/2;
fclose(file);
return;