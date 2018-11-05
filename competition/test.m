function ann
%path='C:\Users\kartik\Documents\MATLAB\steering\';
clear all;
%path = 'F:\steering';
path='/home/kartik/workspace/machine_learning/Assignment3/l3-test/';
data='test-data.txt';
data_path =     fullfile(path,data);
fileID = fopen(data_path,'r');

c=textscan(fileID,'%s');
path='/home/kartik/workspace/machine_learning/Assignment3/l3-test/test/';
images=c{1};
addpath(path);
N=size(c{1});
load('part2_2_32.mat');
ann_test(images, N, w1,w2,w3)



function test_error = ann_test(test, test_size, W1, W2, W3)
    file = fopen('test','a');
    for i=1:test_size
        dataset=test(i,:);
        name = strsplit(dataset{1}, './');
        name = char(name(2));
        x=imread(name);
       
        x=rgb2gray(x);
        x=reshape(x,[1024,1]);
        x=double(x);
        x=x/max(x);
        
        x1=1./(1.+exp(-W1*[1; x]));
        x2=1./(1.+exp(-W2*[1; x1]));
        x3=W3*[1;x2];
     
        fprintf(file,'%s  %f\n',name, x3);
        save('test.mat','name','x3');
    end
return; 


