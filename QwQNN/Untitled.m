a=xlsread('.\result.csv');
%b=xlsread('.\result.csv');
%c=xlsread('.\result.csv');
v=a(:,4);
z=a(:,3);
x=a(:,1);
y=a(:,2);
z=v-z;
figure;
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');%��ֵ
pcolor(X,Y,Z);shading interp%α��ɫͼ
figure,contourf(X,Y,Z) %�ȸ���ͼ
figure,surf(X,Y,Z)%��ά����
shading interp;
