%% Plot 1
t = 0:0.01:15;
st1 = sin(t);
ct1= cos(t);
st2 = 0.9*sin(t);
ct2= 0.9*cos(t);
figure (1);
plot3(st1,ct1,t,st2,ct2,t)
grid on ;
%% Plot 2
t = 0:0.001:15 ;
xt = (1+0.25*cos(200*t)).*cos(t) ;
yt = (1+0.25*cos(200*t)).*sin(t) ;
zt = 2*t+2*sin(200*t) ;
figure (2) ;
plot3(xt,yt,zt) 
grid on ;
%% Plot 3
t = 0:0.001:15 ;
xt = (1+0.25*cos(200*t)).*cos(t) ;
yt = (1+0.25*cos(200*t)).*sin(t) ;
zt = 2*t+2*sin(200*t) ;
xt2 = 0.6*(1+0.25*cos(200*t)).*cos(t) ;
yt2 = 0.6*(1+0.25*cos(200*t)).*sin(t) ;
figure (3) ;
plot3(xt,yt,zt,xt2,yt2,zt) 
grid on ;