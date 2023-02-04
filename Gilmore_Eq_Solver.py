"""
Gilmore equation solver by Youngbin LIM
Contact: lyb0684@naver.com

Input
    dt : time increment
    Total : total time period
    R0 : Initial bubble radius
    pg : initial gas pressure inside bubble
    pext : external pressure
    d : distance for pressure measurement

Output 
    R : time vs bubble radius
    p_wall : bubble radius vs pressure at bubble wall
    pressure contour images
"""
import numpy as np
import matplotlib.pyplot as plt

#Input parameter, all dimesion in m, kg, sec

dt = 1E-10
Total = 0.12E-6
R0 = 10E-6 
pg = 101000.
pext = 10100000.0
d = 1.5*R0 

#Physical constant definition, all dimesion in m, kg, sec
A=300.6993E6; #Constant A for Tait equation of state : A*(Rho/Rho0)^n - B
B=300.598E6; #Constant B for Tait equation of state 
n=7.15; #Exponent n for Tait equation of state 
c0=1480; #Speed of sound in water 
Sigma=0.071; #Surface tension of water
Rho=998.; #Density of water
Gamma=4/3; #Density of water
pv=2340; #Cavitation limit pressure
u=0.001002; #Viscosity of water 

N_total = int(Total/dt) # Number of total increment
t = np.linspace(0., Total, N_total, endpoint=True); #time
R=np.zeros(N_total); #List for bubble radius
dR=np.zeros(N_total); #List for velocity of bubble wall
d2R=np.zeros(N_total); #List for acceleration of bubble wall
H=np.zeros(N_total); #List for enthalpy
p_wall=np.zeros(N_total); #List for pressure at bubble wall
p_shock=np.zeros(N_total); #List for shock pressure measured at d

R[0] = R0; dR[0] = 0 #Initial velocity of bubble wall is assumed to be zero

#Runge-Kutta 6th order method is used to solve ODE for bubble radius with time

for i in range(N_total-1):
    ###########
    # Point 1 #
    ###########
    h1=((n/(n-1))*(A**(1/n))/Rho)*((((pg+(2*Sigma/R[i]))*((R0/R[i])**(3*Gamma))+pv-(2*Sigma/R[i])-4*u*(dR[i]/R[i])+B)**((n-1)/n))-(pext+B)**((n-1)/n)); #Enthalpy   
    c1=((c0**2)+(n-1)*h1)**(0.5); #Speed of sound
    Zeta1=((1+(dR[i]/c1))*h1-1.5*(1-(dR[i]/(3*c1)))*(dR[i]*dR[i]))/(R[i]*(1-(dR[i]/c1)));
    Alpha1=((pg+(2*Sigma/R[i]))*((R0/R[i])**(3*Gamma))+pv-(2*Sigma/R[i])-4*u*(dR[i]/R[i])+B)**(-1/n);
    Beta1=(-2*Sigma/(R[i]*R[i]))*((R0/R[i])**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R[i]))*((R0/R[i])**(3*Gamma))*(1/R[i])+(2*Sigma/(R[i]*R[i]))+4*u*(dR[i]/(R[i]*R[i]));
    d2R1= (Zeta1+(dR[i]/c1)*((A**(1/n))/(Rho))*Alpha1*Beta1)/(1+4*u*((A**(1/n))/(Rho))*Alpha1/(c1*R[i]));
    
    R2=R[i]+(dR[i]*(dt/4));
    dR2=dR[i]+(d2R1*(dt/4));
    
    ###########
    # Point 2 #
    ###########
    h2=((n/(n-1))*(A**(1/n))/Rho)*((((pg+(2*Sigma/R2))*((R0/R2)**(3*Gamma))+pv-(2*Sigma/R2)-4*u*(dR2/R2)+B)**((n-1)/n))-(pext+B)**((n-1)/n));
    c2=((c0**2)+(n-1)*h2)**(0.5);
    Zeta2=((1+(dR2/c2))*h2-1.5*(1-(dR2/(3*c2)))*(dR2*dR2))/(R2*(1-(dR2/c2)));
    Alpha2=((pg+(2*Sigma/R2))*((R0/R2)**(3*Gamma))+pv-(2*Sigma/R2)-4*u*(dR2/R2)+B)**(-1/n);
    Beta2=(-2*Sigma/(R2*R2))*((R0/R2)**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R2))*((R0/R2)**(3*Gamma))*(1/R2)+(2*Sigma/(R2*R2))+4*u*(dR2/(R2*R2));
    d2R2=(Zeta2+(dR2/c2)*((A**(1/n))/(Rho))*Alpha2*Beta2)/(1+4*u*((A**(1/n))/(Rho))*Alpha2/(c2*R2));
    
    R3=R[i]+(3/32)*(dR[i]+3*dR2)*dt;
    dR3=dR[i]+(3/32)*(d2R1+3*d2R2)*dt;
   
    ###########
    # Point 3 #
    ###########
    h3=((n/(n-1))*(A**(1/n))/Rho)*(((pg+2*Sigma/R3)*((R0/R3)**(3*Gamma))-(2*Sigma/R3)+B)**((n-1)/n)-(pext+B)**((n-1)/n)); 
    c3=((c0**2)+(n-1)*h3)**(0.5); 
    Zeta3=((1+(dR3/c3))*h3-1.5*(1-(dR3/(3*c3)))*(dR3*dR3))/(R3*(1-(dR3/c3)));
    Alpha3=((pg+(2*Sigma/R3))*((R0/R3)**(3*Gamma))+pv-(2*Sigma/R3)-4*u*(dR3/R3)+B)**(-1/n);
    Beta3=(-2*Sigma/(R3*R3))*((R0/R3)**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R3))*((R0/R3)**(3*Gamma))*(1/R3)+(2*Sigma/(R3*R3))+4*u*(dR3/(R3*R3));
    d2R3=(Zeta3+(dR3/c3)*((A**(1/n))/(Rho))*Alpha3*Beta3)/(1+4*u*((A**(1/n))/(Rho))*Alpha3/(c3*R3));
    
    R4=R[i]+(12/2197)*(161*dR[i]-600*dR2+608*dR3)*dt ;
    dR4=dR[i]+(12/2197)*(161*d2R1-600*d2R2+608*d2R3)*dt;
    
    ###########
    # Point 4 #
    ###########   
    h4=((n/(n-1))*(A**(1/n))/Rho)*(((pg+2*Sigma/R4)*((R0/R4)**(3*Gamma))-(2*Sigma/R4)+B)**((n-1)/n)-(pext+B)**((n-1)/n));
    c4=((c0**2)+(n-1)*h4)**(0.5);
    Zeta4=((1+(dR4/c4))*h4-1.5*(1-(dR4/(3*c4)))*(dR4*dR4))/(R4*(1-(dR4/c4)));
    Alpha4=((pg+(2*Sigma/R4))*((R0/R4)**(3*Gamma))+pv-(2*Sigma/R4)-4*u*(dR4/R4)+B)**(-1/n);
    Beta4=(-2*Sigma/(R4*R4))*((R0/R4)**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R4))*((R0/R4)**(3*Gamma))*(1/R4)+(2*Sigma/(R4*R4))+4*u*(dR4/(R4*R4));
    d2R4=(Zeta4+(dR4/c4)*((A**(1/n))/(Rho))*Alpha4*Beta4)/(1+4*u*((A**(1/n))/(Rho))*Alpha4/(c4*R4));

    R5=R[i]+(1/4104)*(8341*dR[i]-32832*dR2+29440*dR3-845*dR4)*dt;
    dR5=dR[i]+(1/4104)*(8341*d2R1-32832*d2R2+29440*d2R3-845*d2R4)*dt;
    
    ###########
    # Point 5 #
    ########### 
    h5=((n/(n-1))*(A**(1/n))/Rho)*(((pg+2*Sigma/R5)*((R0/R5)**(3*Gamma))-(2*Sigma/R5)+B)**((n-1)/n)-(pext+B)**((n-1)/n));
    c5=((c0**2)+(n-1)*h5)**(0.5);
    Zeta5=((1+(dR5/c5))*h5-1.5*(1-(dR5/(3*c5)))*(dR5*dR5))/(R5*(1-(dR5/c5)));
    Alpha5=((pg+(2*Sigma/R5))*((R0/R5)**(3*Gamma))+pv-(2*Sigma/R5)-4*u*(dR5/R5)+B)**(-1/n);
    Beta5=(-2*Sigma/(R5*R5))*((R0/R5)**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R5))*((R0/R5)**(3*Gamma))*(1/R5)+(2*Sigma/(R5*R5))+4*u*(dR5/(R5*R5));
    d2R5=(Zeta5+(dR5/c5)*((A**(1/n))/(Rho))*Alpha5*Beta5)/(1+4*u*((A**(1/n))/(Rho))*Alpha5/(c5*R5));
    
    R6=R[i]+(-(8/27)*dR[i]+2*dR2-(3544/2565)*dR3+(1859/4104)*dR4-(11/40)*dR5)*dt;
    dR6=dR[i]+(-(8/27)*d2R1+2*d2R2-(3544/2565)*d2R3+(1859/4104)*d2R4-(11/40)*d2R5)*dt;       
    
    ###########
    # Point 6 #
    ###########     
    h6=((n/(n-1))*(A**(1/n))/Rho)*(((pg+2*Sigma/R6)*((R0/R6)**(3*Gamma))-(2*Sigma/R6)+B)**((n-1)/n)-(pext+B)**((n-1)/n));
    c6=((c0**2)+(n-1)*h6)**(0.5);
    Zeta6=((1+(dR6/c6))*h6-1.5*(1-(dR6/(3*c6)))*(dR6*dR6))/(R6*(1-(dR6/c6)));
    Alpha6=((pg+(2*Sigma/R6))*((R0/R6)**(3*Gamma))+pv-(2*Sigma/R6)-4*u*(dR6/R6)+B)**(-1/n);
    Beta6=(-2*Sigma/(R6*R6))*((R0/R6)**(3*Gamma))-3*Gamma*(pg+(2*Sigma/R6))*((R0/R6)**(3*Gamma))*(1/R6)+(2*Sigma/(R6*R6))+4*u*(dR6/(R6*R6));
    d2R6=(Zeta6+(dR6/c6)*((A**(1/n))/(Rho))*Alpha6*Beta6)/(1+4*u*((A**(1/n))/(Rho))*Alpha6/(c6*R6)); 
    
    k=(1/5)*((16/27)*dR[i]+(6656/2565)*dR3+(28561/11286)*dR4-(9/10)*dR5+(2/11)*dR6);
    kdot=(1/5)*((16/27)*d2R1+(6656/2565)*d2R3+(28561/11286)*d2R4-(9/10)*d2R5+(2/11)*d2R6);
    
    r=R[i] + k*dt;
    dr=dR[i] + kdot*dt; 
    
    R[i+1]=r; dR[i+1]=dr; d2R[i+1]=d2R1; H[i+1]=h1;


#Pressure at bubble wall
for i in range(N_total-1):
    p_wall[i]=((pg+(2*Sigma/R[i]))*((R0/R[i])**(3*Gamma))+pv-(2*Sigma/R[i])-4*u*(dR[i]/R[i]));

#Pressure at distance d
for i in range(N_total-1):
    G=R[i]*(H[i]+((dR[i]*dR[i])/2));
    p_shock[i]=A*(((2/(n+1))+((n-1)/(n+1))*((1+((n+1)*G/(d*c0*c0)))**(1/2)))**(2*n/(n-1)))-B;


#For animation 
Scale=3.; #Size of domain for pressure contour, R0 x Scale
Number_of_spatial_points = 100;
x=np.linspace(-Scale*R0,Scale*R0, Number_of_spatial_points, endpoint=True);   
y=np.linspace(-Scale*R0,Scale*R0, Number_of_spatial_points, endpoint=True);
X, Y = np.meshgrid(1000000*x, 1000000*y)
frame_num=20
levels=np.linspace(-100,max(0.000001*p_wall),200, endpoint=True);
frame=np.linspace(1,frame_num,frame_num, endpoint=True);
p_contour=np.zeros((len(y),len(x))) #Pressure distribution
level_set=np.zeros((len(y),len(x))) #Level set function
inc=int((len(R)+1)/frame_num)

for k in range(len(frame)):
    for i in range(len(x)):
        for j in range(len(y)):
            if np.sqrt(x[i]*x[i]+y[j]*y[j])<=R[k*inc]:
                p_contour[j,i]=((pg+(2*Sigma/R[k*inc]))*((R0/R[k*inc])**(3*Gamma))+pv-(2*Sigma/R[k*inc])-4*u*(dR[k*inc]/R[k*inc]));
                level_set[j,i]=1
            elif np.sqrt(x[i]*x[i]+y[j]*y[j])>R[k*inc]:
                r=np.sqrt(x[i]*x[i]+y[j]*y[j]);
                G=R[k*inc]*(H[k*inc]+((dR[k*inc]*dR[k*inc])/2));
                p_contour[j,i]=A*(((2/(n+1))+((n-1)/(n+1))*((1+((n+1)*G/(r*c0*c0)))**(1/2)))**(2*n/(n-1)))-B;
                level_set[j,i]=0
                
    p_contour=0.000001*p_contour
    time = 1000000*t[k*inc]
    p_max=np.max(p_contour)
    #Pressure contour
    title='time='+str(time)[0:6]+' micro sec'+',  max pressure='+str(p_max)[0:5]+' MPa'
    plt.contourf(X,Y,p_contour, levels, cmap='jet')
    plt.contour(X,Y,level_set, 1, cmap='Reds')
    plt.xlabel('x-distance (micro meter)')
    plt.ylabel('y-distance (micro meter)')
    plt.title(title)
    file_name='For_GIF'+'\\Contour\\frame_' + str(k) + '.png'
    plt.savefig(file_name)
    plt.show()
     
#Plot graphs
   
R=1000000.*R #Bubble radius to micro meter
p_wall=0.000001*p_wall #Pa to MPa
p_shock=0.000001*p_shock #Pa to MPa
d = 1000000.*d 

#Import benchmark data
file=open("Nagrath_radius.txt")
Reference_radius=np.loadtxt(file, delimiter=",");
file.close()

file=open("Nagrath_Wall_Pressure.txt")
Reference_wall=np.loadtxt(file, delimiter=",");
file.close()

#Time vs Bubble radius
for i in range(len(frame)):
    time = 1000000*t[i*inc]
    time_radius= 1000000*t
    Vertical_Bar=np.zeros((2,2));
    Vertical_Bar[0,0]=time; Vertical_Bar[0,1]=0; Vertical_Bar[1,0]=time; Vertical_Bar[1,1]=20.
    plt.plot(time_radius,R)
    plt.plot(Vertical_Bar[:,0], Vertical_Bar[:,1], '--')
    plt.plot(Reference_radius[:,0], Reference_radius[:,1], color='grey', marker='o', linestyle='')
    plt.xlabel('time (micro second)')
    plt.ylabel('bubble radius (micro meter)')
    plt.xlim([0.,0.12])    
    plt.ylim([0.,12.])
    file_name='For_GIF'+'\\Radius\\frame_' + str(i) + '.png'
    plt.savefig(file_name)
    plt.show()
   
    
#Bubble radius vs pressure at bubble wall
for i in range(len(frame)):
    Vertical_Bar=np.zeros((2,2));
    Vertical_Bar[0,0]=R[i*inc]; Vertical_Bar[0,1]=-50; Vertical_Bar[1,0]=R[i*inc]; Vertical_Bar[1,1]=900.
   
    plt.plot(R,p_wall)
    plt.plot(Vertical_Bar[:,0], Vertical_Bar[:,1], '--')
    plt.plot(Reference_wall[:,0], Reference_wall[:,1], color='grey', marker='o', linestyle='')
    plt.xlabel('bubble radius (micro meter)')
    plt.ylabel('bubble wall pressure (MPa)')
   
    plt.xlim([1.,10.])
    plt.ylim([-50,800])
    file_name='For_GIF'+'\\Wall_pressure\\frame_' + str(i) + '.png'
    plt.savefig(file_name)
    plt.show()    

#Shock wave measured at distance d
for i in range(len(frame)):
    time = 1000000*t[i*inc]
    time_shock= 1000000*t
    Vertical_Bar=np.zeros((2,2));
    Vertical_Bar[0,0]=time; Vertical_Bar[0,1]=1.05*min(p_shock); Vertical_Bar[1,0]=time; Vertical_Bar[1,1]=1.05*max(p_shock)
    #Pressure 
    plt.plot(time_shock, p_shock)
    plt.plot(Vertical_Bar[:,0], Vertical_Bar[:,1], '--')    
    plt.xlabel('time (micro second)')
    plt.ylabel('shock pressure (MPa)')
    plt.xlim([0.,0.12])    
    plt.ylim([1.05*min(p_shock),1.05*max(p_shock)])    
    file_name='For_GIF'+'\\Shock\\frame_' + str(i) + '.png'
    plt.savefig(file_name)
    plt.show()    