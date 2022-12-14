a=xlsread('.\result.csv');
%b=xlsread('.\result.csv');
%c=xlsread('.\result.csv');
v=a(:,4);
z=a(:,3);
x=a(:,1);
y=a(:,2);
z=v-z;
figure;
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');%插值
pcolor(X,Y,Z);shading interp%伪彩色图
figure,contourf(X,Y,Z) %等高线图
figure,surf(X,Y,Z)%三维曲面
shading interp;
