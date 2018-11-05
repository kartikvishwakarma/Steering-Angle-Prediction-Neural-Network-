function ann
%path='C:\Users\kartik\Documents\MATLAB\steering\';
clear all;
path = 'F:\steering';
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


w1=-0.3+(0.6)*rand(512,(1024+1));
w2=-0.3+(0.6)*rand(64,(512+1));
w3=-0.3+(0.6)*rand(1,(64+1));

%train_size=size(train,1);
%test_size=size(test,1);

disp('train function Called');
[W1, W2, W3, ERROR]= ann_train(train, valid, w1, w2, w3);
disp('train function Returned');

%disp(ERROR);
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
        %disp(error/2);
        disp('Error!!!');
        test_error = error/2;
        %file_10=fopen('validation_error.txt','w');
        %fprintf(file_10,'%s %d\n', error/2,i);
        dlmwrite('test_error.txt', error/2, 'delimiter','\t');
return; 



function [ w1, w2, w3, error ] = ann_train(train, valid, w1, w2, w3)

epochs=1000;
eta=0.001;
prob=0.5;
batch_size=64;
train_size=size(train,1);
valid_size=size(valid,1);
E=0;
error=0;
x=[];
y=[];
v_x=[];
v_y=[];
dropout1=[];
dropout2=[];
dropout3=[];
file=fopen('part2_3_dropout_error.txt','a');

for i=1:epochs
%--------------------validation_model-------------------------------%    
    v_error=[];
    for vk=1:batch_size:floor(valid_size/batch_size)*batch_size
        %zero=prob*1024;
        %map1=[zeros(1,zero) ones(1,1024-zero)];
        %map1=map1(randperm(1024));
    
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
        
        %dropout1=repmat(map1,batch_size,1);
        %xx=x.*dropout1';
        
        v_x1=1./(1.+exp(-w1*[ones(1,batch_size); v_x]));
        
        %zero=prob*128;
        %map2=[zeros(1,zero) ones(1,128-zero)];
        %map2=map2(randperm(128));
    
       
        %dropout2=repmat(map2,batch_size,1);
        
        %xx1=x1.*dropout2';
        v_x2=1./(1.+exp(-w2*[ones(1,batch_size); v_x1]));
        
        %zero=prob*64;
        %map3=[zeros(1,zero) ones(1,64-zero)];
        %map3=map3(randperm(64));
        
       
        %dropout3=repmat(map3,batch_size,1);
       
        %xx2=x2.*dropout3';
        
        v_x3=w3*[ones(1,batch_size);v_x2];
        E=(v_x3-v_y).*(v_x3-v_y);
        v_error=[v_error sum(E)];
        v_x=[];
        v_y=[];
    end
%--------------------train_model------------------------------------%    
    error=[];
    for k=1:batch_size:floor(train_size/batch_size)*batch_size
        zero=prob*1024;
        map1=[zeros(1,zero) ones(1,1024-zero)];
        map1=map1(randperm(1024));
    
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
        
        dropout1=repmat(map1,batch_size,1);
        xx=x.*dropout1';
    
        x1=1./(1.+exp(-w1*[ones(1,batch_size); xx]));
        
        zero=prob*512;
        map2=[zeros(1,zero) ones(1,512-zero)];
        map2=map2(randperm(512));
    
       
        dropout2=repmat(map2,batch_size,1);
        
        xx1=x1.*dropout2';
        x2=1./(1.+exp(-w2*[ones(1,batch_size); xx1]));
        
        zero=prob*64;
        map3=[zeros(1,zero) ones(1,64-zero)];
        map3=map3(randperm(64));
        
       
        dropout3=repmat(map3,batch_size,1);
       
        xx2=x2.*dropout3';
        
        x3=w3*[ones(1,batch_size);xx2];
        
        delta2=(x3-y);
        dE3=delta2*[ones(1,batch_size);xx2]';
        
        delta1 = (w3(:,2:end)'*delta2).*(x2.*(1-x2));
        dE2 = delta1*[ones(1,batch_size);xx1]';
        
        delta0 = (w2(:,2:end)'*delta1).*(x1.*(1-x1));
        dE1 = delta0*[ones(1,batch_size);xx]';

        w3=w3-eta*dE3;
        w2=w2-eta*dE2;
        w1=w1-eta*dE1;
    
        E=((x3-y).*(x3-y));
        
       % disp(sum(E));
    %error=error+E;
    
        x=[];
        y=[];
    
        dropout1=[];
        dropout2=[];
        dropout3=[];
        
        error=[error; sum(E)];
       % sprintf('Error...%f',E)
       %  sprintf('Sum of error...%f',sum(E))
    end
    sprintf('Epochs......%d',i)
    %if(mod(i,100)==0)
    %disp(error);
    %disp(train_size);
    fprintf(file,'%f        %f      %d\n', sum(error)/(2*train_size),sum(v_error)/(2*valid_size),i);
   
end
w1=w1;
w2=w2;
w3=w3;
error=E/2;
save('part2_3_dropout.mat','w1','w2','w3');
fclose(file);
return;