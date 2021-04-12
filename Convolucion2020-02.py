import streamlit as st
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats 
import numpy as np
import pylab as pl
import webbrowser

def ConvContinuo(tx,x,th,h):
        st.write(""" **Convolución** """)
        empty = st.empty()

        espacio=max(-th)-min(tx)
        t2inv=(-th-espacio)
        frame = 30

        y1=signal.convolve(x,h)

        tconv=np.linspace( min(tx), max(tx)+(max(th)-min(th)), len(y1) )

        for  i in range( 0,(frame+1) ):      

                fig=plt.figure(figsize=(12,10))
                fig.tight_layout()

                ax3 = fig.add_subplot(2,1,1)
                ax3.clear()
                ax3.set_title('Gráfica animacion y[n]=x[n]*h[n]')
                ax3.set_xlabel('t[n]')
                ax3.set_ylabel('x[n] y h[n]')         
                plt.grid(b=True,which='major')
                plt.minorticks_on()
                plt.grid(b=True,which='minor',linestyle='-',alpha=0.3) 
                plt.xlim( min(-th) - (max(-th)-min(tx)) , max(tx) + (max(th)-min(th)) )
                ax3.plot( t2inv+( ((max(tx)-min(t2inv))/frame) *i),h,color=colorax2) 
                ax3.plot(tx,x,color=colorax1)  
                ax3.legend(['h(t)','x(t)'],loc='upper left')
                                    
                pulse=np.piecewise( tconv, [tconv<( min(tx)+( ((max(tx)-min(t2inv))/frame) *i) )\
                    , tconv>=(min(tx)+( ((max(tx)-min(t2inv))/frame) *i) )],[1,0])
                y=pulse*y1

                ax4 = fig.add_subplot(2,1,2)
                ax4.clear()
                colorax4='tab:brown'
                ax4.set_title('Gráfica convolucion y[n]=x[n]*h[n]')
                ax4.set_xlabel('t[s]')
                ax4.set_ylabel('y[t]')
                plt.xlim( min(-th) - (max(-th)-min(tx)) , max(tx) + (max(th)-min(th)))
                plt.grid(b=True,which='major')
                plt.minorticks_on()
                plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)
                ax4.plot(tconv,y,color=colorax4)    
                ax4.legend(['y(t)'],loc='upper left')

                st.set_option('deprecation.showPyplotGlobalUse', False)
                empty.pyplot(fig)  

def ConvDiscreto(tx,x,th,h):
        st.write(""" **Convolución** """)
        empty = st.empty()

        espacio=max(-th)-min(tx)
        t2inv=(-th-espacio)
        frame =20

        y1=signal.convolve(x,h)

        tconv=np.arange( int(min(tx)), int(max(tx)+(max(th)-min(th)))+1, 1 )

        for  i in range( 0,(frame+1) ):   

                fig=plt.figure(figsize=(12,10))
                fig.tight_layout()
                
                ax3 = fig.add_subplot(2,1,1)
                ax3.clear()
                ax3.set_title('Gráfica animacion y[n]=x[n]*h[n]')
                ax3.set_xlabel('t[n]')
                ax3.set_ylabel('x[n] y h[n]')         
                plt.grid(b=True,which='major')
                plt.minorticks_on()
                plt.grid(b=True,which='minor',linestyle='-',alpha=0.3) 
                plt.xlim( min(-th) - (max(-th)-min(tx)) , max(tx) + (max(th)-min(th)) )
                ax3.stem( t2inv+( ((max(tx)-min(t2inv))/frame) *i),h) 
                ax3.stem(tx,x)  
                ax3.legend(['h[n]','x[n]'],loc='upper left')
                        
                pulse=np.piecewise( tconv, [tconv<( min(tx)+( ((max(tx)-min(t2inv))/frame) *i) )\
                    , tconv>=(min(tx)+( ((max(tx)-min(t2inv))/frame) *i) )],[1,0])
                y=pulse*y1

                ax4 = fig.add_subplot(2,1,2)
                ax4.clear()
                colorax4='tab:brown'
                ax4.set_title('Gráfica convolucion y[n]=x[n]*h[n]')
                ax4.set_xlabel('t[s]')
                ax4.set_ylabel('y[t]')
                plt.xlim( min(-th) - (max(-th)-min(tx)) , max(tx) + (max(th)-min(th)))
                plt.grid(b=True,which='major')
                plt.minorticks_on()
                plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)
                ax4.stem(tconv,y)    
                ax4.legend(['y([t]'],loc='upper left')
                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                empty.pyplot(fig) 

def RepFourier(tx,x,th,h,tinicialx,tfinalx,n):
        
    colorax1 = 'tab:red'
    colorax2 = 'tab:blue'

    fig=plt.figure(figsize=(12,20))
    fig.tight_layout()

    ax5 = fig.add_subplot(3,1,1)
    ax5.set_title('Gráfica X(f)')
    colorax1 = 'tab:red'     
    ax5.set_xlabel('f(w)', color=colorax1)
    ax5.set_ylabel('X(f)', color=colorax1)  
    plt.xlim(tinicialx,tfinalx)
    plt.grid(b=True,which='major')
    plt.minorticks_on()
    plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)

    ax6 = fig.add_subplot(3,1,2)
    ax6.set_title('Gráfica H(f)')
    ax6.set_ylabel('X(f)', color=colorax1)  
    colorax2 = 'tab:blue'
    ax6.set_xlabel('f(w)', color=colorax1)
    ax6.set_ylabel('H(f)', color=colorax1) 
    plt.xlim(tinicialx,tfinalx)
    plt.grid(b=True,which='major')
    plt.minorticks_on()
    plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)

    ax7 = fig.add_subplot(3,1,3)
    ax7.set_title('Gráfica H(f)')
    ax7.set_ylabel('X(f)', color=colorax1)  
    colorax2 = 'tab:blue'
    ax7.set_xlabel('f(w)', color=colorax1)
    ax7.set_ylabel('H(f)', color=colorax1) 
    plt.xlim(tinicialx,tfinalx)
    plt.grid(b=True,which='major')
    plt.minorticks_on()
    plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)
    
    dt = 0.01
    T=tfinalx-tinicialx
    wo = (2*np.pi)/T
    ak = np.zeros(n)
    bk = np.zeros(n)
    m = len(tx)
    A0 = 0
    for i in range(1,m):
        A0 = A0 +(1/T)*x[i]*dt
    for i in range(1,n):
        for j in range(0,int(m)):
            ak[i] = ak[i] + ((2/T)*x[j]*np.cos(i*tx[j]*wo))*dt
            bk[i] = bk[i] + ((2/T)*x[j]*np.sin(i*tx[j]*wo))*dt

    r = 0.1
    t1 = np.arange(0, 10+r, r)
    xf= 0*t1+A0

    for i in range(1, n):
        xf = xf + ak[i]*np.cos(i*wo*t1)+bk[i]*np.sin(i*wo*t1)

    n_wo = np.arange(n)
    ck = np.zeros(n)
    phik = np.zeros(n)

    for i in range(n):
        ck[i] = (ak[i]**2+bk[i]**2)**0.5
        phik[i] = -np.arctan(bk[i]/ak[i])

    ax5.plot(t1,xf,color = colorax1)
    ax5.legend(['x(t)'],loc='upper left')
    ax6.stem(n_wo,ck)
    ax6.legend(['x(t)'],loc='upper left')
    ax7.stem(n_wo,phik)
    ax7.legend(['x(t)'],loc='upper left')

    st.pyplot(fig)

st.write(""" 
# Segundo laboratorio de Señales y Sistemas

Codigo realizado por **Dierick Brochero Niebles**

""")
Tiempo= st.radio("Selecione la representación en el tiempo:",('Continuo','Discreto'))

Signal1 = st.selectbox("Seleccione la función x(t)",["Seleccione una función"\
    ,"Coseno","Seno","Pulso","Exponencial","Cuadrática","Lineal","Triangular",\
    "Cuadrada","Señal rampa 1","Señal rampa 2","Señal rampa 3"], index=0)
st.sidebar.title('Parametros señal x(t)')
Amplitud1 = st.sidebar.slider("Amplitud de la señal x(t) ",min_value=-10, max_value=10, value=1, step=1)
tinicialx=st.sidebar.number_input("Ingrese el t inicial de la señal x(t)", value=0.0, step=0.1)   
tfinalx = st.sidebar.number_input("'Ingrese el t final de la señal x(t)", value=0.0, step=0.1)
EscTiem1 =st.sidebar.number_input("Ingrese el Escalamiento del tiempo de la señal x(t)", value=1.0, step=0.1) 
DespTiemp1 =st.sidebar.number_input("Ingrese el desplazamiento del tiempo de la señal x(t)", value=0.0, step=0.1) 

Signal2 = st.selectbox("Seleccione la función h(t)",["Seleccione una función"\
    ,"Seno","Coseno","Pulso","Exponencial","Cuadrática","Lineal","Triangular",\
    "Cuadrada","Señal rampa 1","Señal rampa 2","Señal rampa 3"], index=0)
st.sidebar.title('Parametros señal h(t)')
Amplitud2= st.sidebar.slider("Amplitud de la señal h(t) ",min_value=-10, max_value=10, value=1, step=1)
tinicialh=st.sidebar.number_input("Ingrese el t inicial de la señal h(t)",value=0.0, step=0.1)   
tfinalh = st.sidebar.number_input("'Ingrese el t final de la señal h(t)",value=0.0, step=0.1)
EscTiem2 =st.sidebar.number_input("Ingrese el Escalamiento del tiempo de la señal h(t)", value=1.0, step=0.1) 
DespTiemp2 =st.sidebar.number_input("Ingrese el desplazamiento del tiempo de la señal h(t)",value=0.0, step=0.1) 

st.sidebar.title('Número de armonicos para la representación en series de Fourier')
n=st.sidebar.number_input("Ingrese el número n de armonicos", value=0, step=1)

Codigo=st.button('Mostrar código')
if Codigo:
    webbrowser.open(
        'https://github.com/Dierickb/Signals-and-Systmes/blob/Mediante-funciones/Convolucion2020-02.py'
    )

Convolucion=st.button("Gráficar Convolución")
Fourier=st.button("Representación mediante series de Fourier")

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8)
fig=plt.figure(figsize=(12,10))
fig.tight_layout()

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Gráfica x(t)')
colorax1 = 'tab:red'     
ax1.set_xlabel('t(s)', color=colorax1)
ax1.set_ylabel('x(t)', color=colorax1)  
plt.xlim(tinicialx,tfinalx)
plt.grid(b=True,which='major')
plt.minorticks_on()
plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Gráfica h(t)')
colorax2 = 'tab:blue'
ax2.set_xlabel('t(s)', color=colorax2)
ax2.set_ylabel('h(t)', color=colorax2) 
plt.xlim(tinicialh,tfinalh)
plt.grid(b=True,which='major')
plt.minorticks_on()
plt.grid(b=True,which='minor',linestyle='-',alpha=0.3)

if Tiempo=='Continuo':
    #------------x(t)---------#
    tamtx=(tfinalx-tinicialx)/0.01
    tx = np.linspace(tinicialx,tfinalx, int(tamtx) )
    if   Signal1 == "Coseno":     
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 

        ax1.set_title('Gráfica x(t)=Amp*cos((t*EscTiem)+DespTiemp)') 
        x = Amplitud1*np.cos((EscTiem1*tx*frecuencia1)+DespTiemp1) 
        ax1.plot(tx,x,color = colorax1)
        ax1.legend(['x(t)'],loc='upper left')

    elif Signal1 == "Seno":       
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 
        
        ax1.set_title('Gráfica x(t)=Amp*sin((t*EscTiem)+DespTiemp)')
        x = Amplitud1*np.sin((EscTiem1*tx*frecuencia1)+DespTiemp1)  
        ax1.plot(tx,x,color = colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Exponencial":
        ax1.set_title('Gráfica x(t)=Amp*exp((t*{EscTiem)+DespTiemp)')
        x=Amplitud1*np.exp((EscTiem1*tx)+DespTiemp1)
        ax1.plot(tx,x,color = colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Cuadrática": 
        ValorA=float(st.text_input('Ingrese el Valor A de la señal x(t)',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal x(t)',value='0'))
        ValorC=float(st.text_input('Ingrese el Valor C de la señal x(t)',value='0'))
        
        ax1.set_title('Gráfica x(t)=A(t^2)+Bt+C')
        x=ValorA*pow(Amplitud1*tx,2)+(ValorB*Amplitud1*tx)+ValorC
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Lineal":     
        ValorA=float(st.text_input('Ingrese el Valor A de la señal x(t)',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal x(t)',value='0'))
        
        ax1.set_title('Gráfica x(t)=At+B')
        x=ValorA*(Amplitud1*tx)+ValorB
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Pulso":      
        ValorA=float(st.text_input('Ingrese el ValorA(Ancho del impulso) de la señal x(t)',value='0'))
        
        ax1.set_title('Gráfica x(t)=u(t)-u(t-ValorB)')
        x=np.piecewise((tx*EscTiem1),((tx*EscTiem1)+DespTiemp1)>=0,[Amplitud1*1,0])\
            -np.piecewise((tx*EscTiem1),((tx*EscTiem1)+DespTiemp1-ValorA)>=0,[Amplitud1*1,0])
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Triangular": 
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)    
        
        ax1.set_title('Gráfica x(t)=sawtooth(f*(t*EscTiem))+DespTiemp)')
        x=signal.sawtooth((frecuencia1*(tx*EscTiem1))+DespTiemp1,width=0.5)
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')
        
    elif Signal1 == "Cuadrada":   
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 
        
        ax1.set_title('Gráfica x(t)=sawtooth(f*(t*EscTiem))+DespTiemp)')
        
        x=signal.square((frecuencia1*(tx*EscTiem1))+DespTiemp1)
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')

    elif Signal1 == "Señal rampa 1":
        tamtx1=(3.4-0)/(0.01)
        tamtx2=(6.4-(3.4+0.01))/0.01
        tamtx3=(10-(6.4+0.01))/0.01

        tramp11= np.linspace(0, 3.4, int(tamtx1) )
        tramp12= np.linspace( 3.4+0.01, 6.4, int(tamtx2) )
        tramp13= np.linspace( 6.4+0.01, 10, int(tamtx3) )
        x1=np.zeros( len(tramp11) )
        x2= (3.4/2.99)*tramp12-3.877591973
        x3= 3.4 + (tramp13*0)
        
        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')

    elif Signal1 == "Señal rampa 2":
        tamtx1=(3.7-0)/(0.01)
        tamtx2=(6.7-(3.7+0.01))/0.01
        tamtx3=(10-(6.7+0.01))/0.01

        tramp11= np.linspace(0, 3.7, int(tamtx1) )
        tramp12= np.linspace( 3.7+0.01, 6.7, int(tamtx2) )
        tramp13= np.linspace( 6.7+0.01, 10, int(tamtx3) )
        x1=3.4 + (tramp11*0)
        x2= (-3.4/2.99)*tramp12+( (3.4/2.99)*6.7  )
        x3=np.zeros( len(tramp13) )    
        
        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')

    elif Signal1 == "Señal rampa 3":
        tamtx1=(3.29-0)/(0.01)
        tamtx2=(6.7-(3.29+0.01))/0.01
        tamtx3=(10-(6.7+0.01))/0.01

        tramp11= np.linspace(0, 3.29, int(tamtx1) )
        tramp12= np.linspace( 3.29+0.01, 6.7, int(tamtx2) )
        tramp13= np.linspace( 6.7+0.01, 10, int(tamtx3) )
        x1= (3.4/3.29)*(tramp11)
        x2= 3.4+(tramp12*0) 
        x3= (-3.4/3.29)*tramp13 + (34/3.29)
        
        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )
        ax1.plot(tx,x,color=colorax1)
        ax1.legend(['x(t)'],loc='upper left')

    #------------h(t)---------#
    tamth=(tfinalh-tinicialh)/0.01
    th=np.linspace(tinicialh,tfinalh, int(tamth))  
    if   Signal2 == "Seno":       
        frecuencia2 = st.number_input("frecuencia h(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)

        ax2.set_title('Gráfica h(t)=Amp*cos((t*EscTiem)+DespTiemp)') 
        h = Amplitud2*np.sin((EscTiem2*th*frecuencia2)+DespTiemp2) 
        ax2.plot(th,h,color=colorax2) 
        ax2.legend(['h(t)'],loc='upper left')   

    elif Signal2 == "Coseno":     
        frecuencia2 = st.number_input("frecuencia h(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h(t)=Amp*sin((t*EscTiem)+DespTiemp)') 
        h = Amplitud2*np.cos((EscTiem2*th*frecuencia2)+DespTiemp2)
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Exponencial":
        ax2.set_title('Gráfica h(t)=Amp*exp((t*EscTiem)+DespTiemp)')
        h=Amplitud2*np.exp((EscTiem2*th)+DespTiemp2)   
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Cuadrática": 
        ValorA=float(st.text_input('Ingrese el Valor A de la señal h(t)',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal h(t)',value='0'))
        ValorC=float(st.text_input('Ingrese el Valor C de la señal h(t)',value='0'))
        
        ax2.set_title('Gráfica h(t)=A(t^2)+Bt+C')
        h=ValorA*pow(Amplitud2*th,2)+(ValorB*Amplitud2*th)+ValorC
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Lineal":     
        ValorA=float(st.text_input('Ingrese el Valor A de la señal h(t)',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal h(t)',value='0'))
        
        ax2.set_title('Gráfica h(t)=At+C')
        h=ValorA*(Amplitud2*th)+ValorB
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Pulso":      
        ValorA=float(st.text_input('Ingrese el ValorA(Ancho del impulso) de la señal h(t)',value='0'))

        ax2.set_title('Gráfica h(t)=u((t))-u(t-ValorA)')
        
        h=np.piecewise((th*EscTiem2),((th*EscTiem2)+DespTiemp2)>=0,[Amplitud2*1,0])\
            -np.piecewise((th*EscTiem2),((th*EscTiem2)+DespTiemp2-ValorA)>=0,[Amplitud2*1,0])
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Triangular": 
        frecuencia2 = st.number_input("frecuencia h(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h(t)=sawtooth(f*(t*EscTiem))+DespTiemp)')
        h=signal.sawtooth((frecuencia2*(th*EscTiem2))+DespTiemp2,width=0.5)
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Cuadrada":   
        frecuencia2 = st.number_input("frecuencia h(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h(t)=sawtooth(f*(t*EscTiem))+DespTiemp)')
        
        h=signal.square((frecuencia2*(th*EscTiem2))+DespTiemp2)
        ax2.plot(th,h,color=colorax2)  
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Señal rampa 1":
        tamth1=(3.4-0)/(0.01)
        tamth2=(6.4-(3.4+0.01))/0.01
        tamth3=(10-(6.4+0.01))/0.01

        tramp21= np.linspace(0, 3.4, int(tamth1) )
        tramp22= np.linspace( 3.4+0.01, 6.4, int(tamth2) )
        tramp23= np.linspace( 6.4+0.01, 10, int(tamth3) )
        h1=np.zeros( len(tramp21) )
        h2= (3.4/2.99)*tramp22-3.877591973
        h3= 3.4 + (tramp23*0)
        
        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )
        ax2.plot(th,h,color=colorax2)
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Señal rampa 2":
        tamth1=(3.7-0)/(0.01)
        tamth2=(6.7-(3.7+0.01))/0.01
        tamth3=(10-(6.7+0.01))/0.01

        tramp21= np.linspace(0, 3.7, int(tamth1) )
        tramp22= np.linspace( 3.7+0.01, 6.7, int(tamth2) )
        tramp23= np.linspace( 6.7+0.01, 10, int(tamth3) )
        h1=3.4 + (tramp21*0)
        h2= (-3.4/2.99)*tramp22+( (3.4/2.99)*6.7  )
        h3=np.zeros( len(tramp23) )    
        
        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )
        ax2.plot(th,h,color=colorax2)
        ax2.legend(['h(t)'],loc='upper left')

    elif Signal2 == "Señal rampa 3":
        tamth1=(3.29-0)/(0.01)
        tamth2=(6.7-(3.29+0.01))/0.01
        tamth3=(10-(6.7+0.01))/0.01

        tramp21= np.linspace(0, 3.29, int(tamth1) )
        tramp22= np.linspace( 3.29+0.01, 6.7, int(tamth2) )
        tramp23= np.linspace( 6.7+0.01, 10, int(tamth3) )
        h1= (3.4/3.29)*(tramp21)
        h2= 3.4+(tramp22*0) 
        h3= (-3.4/3.29)*tramp23 + (34/3.29)

        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )
        ax2.plot(th,h,color=colorax2)
        ax2.legend(['h(t)'],loc='upper left')

    st.pyplot(fig)

    if Convolucion:
        ConvContinuo(tx,x, th,h)

if Tiempo=='Discreto':
    #------------x[n]---------#
    tx = np.arange(int(tinicialx),int(tfinalx)+1, 1 )

    if   Signal1 == "Coseno":     
        frecuencia1 = st.number_input("frecuencia x[n]", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 

        ax1.set_title('Gráfica x[n]=Amp*cos[(n*EscTiem)+DespTiemp]') 
        x = Amplitud1*np.cos((EscTiem1*tx*frecuencia1)+DespTiemp1) 
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Seno":       
        frecuencia1 = st.number_input("frecuencia x[n]", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 
        
        ax1.set_title('Gráfica x[n]=Amp*sin[(n*EscTiem)+DespTiemp]')
        x = Amplitud1*np.sin((EscTiem1*tx*frecuencia1)+DespTiemp1)  
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')
            
    elif Signal1 == "Exponencial":
        ax1.set_title('Gráfica x[n]=Amp*exp[(n*{EscTiem)+DespTiemp]')
        x=Amplitud1*np.exp((EscTiem1*tx)+DespTiemp1)
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')
            
    elif Signal1 == "Cuadrática": 
        ValorA=float(st.text_input('Ingrese el Valor A de la señal x(t)',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal x(t)',value='0'))
        ValorC=float(st.text_input('Ingrese el Valor C de la señal x(t)',value='0'))
        
        ax1.set_title('Gráfica x[n]=A(n^2)+Bn+C')
        x=ValorA*pow(Amplitud1*tx,2)+(ValorB*Amplitud1*tx)+ValorC
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Lineal":     
        ValorA=float(st.text_input('Ingrese el Valor A de la señal x[n]',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal x[n]',value='0'))
        
        ax1.set_title('Gráfica x[n]=An+B')
        x=ValorA*(Amplitud1*tx)+ValorB
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')
            
    elif Signal1 == "Pulso":      
        ValorA=float(st.text_input('Ingrese el ValorA(Ancho del impulso) de la señal x[n]',value='0'))
        
        ax1.set_title('Gráfica x[n]=u[n]-u[n-ValorB]')
        x=np.piecewise((tx*EscTiem1),((tx*EscTiem1)+DespTiemp1)>=0,[Amplitud1*1,0])\
            -np.piecewise((tx*EscTiem1),((tx*EscTiem1)+DespTiemp1-ValorA)>=0,[Amplitud1*1,0])
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Triangular": 
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)    
        
        ax1.set_title('Gráfica x[n]=sawtooth[f*(n*EscTiem))+DespTiemp]')
        x=signal.sawtooth((frecuencia1*(tx*EscTiem1))+DespTiemp1,width=0.5)
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Cuadrada":   
        frecuencia1 = st.number_input("frecuencia x(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1) 
        
        ax1.set_title('Gráfica x[n]=sawtooth[f*(n*EscTiem))+DespTiemp]')
        
        x=signal.square((frecuencia1*(tx*EscTiem1))+DespTiemp1)
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Secuencia de impulsos":
        x=(st.text_input('Escriba el tamaño del vector x(t)'))
  
    elif Signal1 == "Señal rampa 1":
        tamtx1=(3.4-0)/(0.01)
        tamtx2=(6.4-(3.4+0.01))/0.01
        tamtx3=(10-(6.4+0.01))/0.01

        tramp11= np.arange(0, int(3.4)+1, 1 )
        tramp12= np.arange( int(3.4+0.01)+1, int(6.4)+1, 1 )
        tramp13= np.arange( int(6.4+0.01)+1, 10+1, 1 )
        x1=np.zeros( len(tramp11) )
        x2= (3.4/2.99)*tramp12-3.877591973
        x3= 3.4 + (tramp13*0)
        
        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Señal rampa 2":
        tamtx1=(3.7-0)/(0.01)
        tamtx2=(6.7-(3.7+0.01))/0.01
        tamtx3=(10-(6.7+0.01))/0.01
        
        tramp11= np.arange(0, int(3.7)+1, 1)
        tramp12= np.arange( int(3.7+0.01)+1, int(6.7)+1, 1) 
        tramp13= np.arange( int(6.7+0.01)+1, 10+1, 1 )

        x1=3.4 + (tramp11*0)
        x2= (-3.4/2.99)*tramp12+( (3.4/2.99)*6.7  )
        x3=np.zeros( len(tramp13) )    
        
        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )        
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    elif Signal1 == "Señal rampa 3":
        tamtx1=(3.29-0)/(0.01)
        tamtx2=(6.7-(3.29+0.01))/0.01
        tamtx3=(10-(6.7+0.01))/0.01

        tramp11= np.arange(0, int(3.29)+1 ,1)
        tramp12= np.arange(int(3.29+0.01)+1, int(6.7)+1, 1)
        tramp13= np.arange(int(6.7+0.01)+1, 10+1, 1) 

        x1= (3.4/3.29)*(tramp11)
        x2= 3.4+(tramp12*0) 
        x3= (-3.4/3.29)*tramp13 + (34/3.29)

        tx= np.concatenate( (tramp11,tramp12,tramp13), axis=None )
        x= np.concatenate( (x1,x2,x3), axis=None )
        ax1.stem(tx,x)
        ax1.legend(['x[n]'],loc='upper left')

    #------------h[n]---------#
    th=np.arange(int(tinicialh),int(tfinalh)+1, 1)  
    if   Signal2 == "Seno":       
        frecuencia2 = st.number_input("frecuencia h[n]", min_value=0.0, max_value=200.0, value=1.0, step=0.1)

        ax2.set_title('Gráfica h[n]=Amp*cos[(n*EscTiem)+DespTiemp]') 
        h = Amplitud2*np.sin((EscTiem2*th*frecuencia2)+DespTiemp2) 
        ax2.stem(th,h) 
        ax2.legend(['h[n]'],loc='upper left')   

    elif Signal2 == "Coseno":     
        frecuencia2 = st.number_input("frecuencia h[n]", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h[n]=Amp*sin[(n*EscTiem)+DespTiemp]') 
        h = Amplitud2*np.cos((EscTiem2*th*frecuencia2)+DespTiemp2)
        ax2.stem(th,h)  
        ax2.legend(['h[n]'],loc='upper left')
            
    elif Signal2 == "Exponencial":
        ax2.set_title('Gráfica h[n]=Amp*exp[(n*EscTiem)+DespTiemp]')
        h=Amplitud2*np.exp((EscTiem2*th)+DespTiemp2)   
        ax2.stem(th,h)  
        ax2.legend(['h[n]'],loc='upper left')
            
    elif Signal2 == "Cuadrática": 
        ValorA=float(st.text_input('Ingrese el Valor A de la señal h[n]',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal h[n]',value='0'))
        ValorC=float(st.text_input('Ingrese el Valor C de la señal h[n]',value='0'))
        
        ax2.set_title('Gráfica h[n]=A(n^2)+Bn+C')
        h=ValorA*pow(Amplitud2*th,2)+(ValorB*Amplitud2*th)+ValorC
        ax2.stem(th,h2)  
        ax2.legend(['h[n]'],loc='upper left')
            
    elif Signal2 == "Lineal":     
        ValorA=float(st.text_input('Ingrese el Valor A de la señal h[n]',value='0'))
        ValorB=float(st.text_input('Ingrese el Valor B de la señal h[n]',value='0'))
        
        ax2.set_title('Gráfica h[n]=An+C')
        h=ValorA*(Amplitud2*th)+ValorB
        ax2.stem(th,h)  
        ax2.legend(['h[n]'],loc='upper left')

    elif Signal2 == "Pulso":      
        ValorA=float(st.text_input('Ingrese el ValorA(Ancho del impulso) de la señal h[n]',value='0'))

        ax2.set_title('Gráfica h[n]=u[n]-u[n-ValorA]')
        
        h=np.piecewise((th*EscTiem2),((th*EscTiem2)+DespTiemp2)>=0,[Amplitud2*1,0])\
            -np.piecewise((th*EscTiem2),((th*EscTiem2)+DespTiemp2-ValorA)>=0,[Amplitud2*1,0])
        ax2.stem(th,h)  
        ax2.legend(['[n]'],loc='upper left')
            
    elif Signal2 == "Triangular": 
        frecuencia2 = st.number_input("frecuencia h[n]", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h[n]=sawtooth[f*(n*EscTiem))+DespTiemp]')
        h=signal.sawtooth((frecuencia2*(th*EscTiem2))+DespTiemp2,width=0.5)
        ax2.stem(th,h)  
        ax2.legend(['h[n]'],loc='upper left')
            
    elif Signal2 == "Cuadrada":   
        frecuencia2 = st.number_input("frecuencia h(t)", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
        
        ax2.set_title('Gráfica h[n]=sawtooth[f*(n*EscTiem))+DespTiemp]')
        
        h=signal.square((frecuencia2*(th*EscTiem2))+DespTiemp2)
        ax2.stem(th,h)  
        ax2.legend(['h[n]'],loc='upper left')

    elif Signal2 == "Señal rampa 1":
        tamth1=(3.4-0)/(0.01)
        tamth2=(6.4-(3.4+0.01))/0.01
        tamth3=(10-(6.4+0.01))/0.01

        tramp21= np.arange(0, int(3.4)+1, 1 )
        tramp22= np.arange( int(3.4+0.01)+1, int(6.4)+1, 1 )
        tramp23= np.arange( int(6.4+0.01)+1, 10+1, 1 )
        h1=np.zeros( len(tramp21) )
        h2= (3.4/2.99)*tramp22-3.877591973
        h3= 3.4 + (tramp23*0)
        
        ax2.set_title('Gráfica h[n]')

        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )
        ax2.stem(th,h)
        ax2.legend(['h[n]'],loc='upper left')

    elif Signal2 == "Señal rampa 2":
        tamth1=(3.7-0)/(0.01)
        tamth2=(6.7-(3.7+0.01))/0.01
        tamth3=(10-(6.7+0.01))/0.01
        
        tramp21= np.arange(0, int(3.7)+1, 1)
        tramp22= np.arange( int(3.7+0.01)+1, int(6.7)+1, 1) 
        tramp23= np.arange( int(6.7+0.01)+1, 10+1, 1 )

        h1=3.4 + (tramp21*0)
        h2= (-3.4/2.99)*tramp22+( (3.4/2.99)*6.7  )
        h3=np.zeros( len(tramp23) )    

        ax2.set_title('Gráfica h[n]')
        
        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )        
        ax2.stem(th,h)
        ax2.legend(['h[n]'],loc='upper left')

    elif Signal2 == "Señal rampa 3": 
        tamth1=(3.29-0)/(0.01)
        tamth2=(6.7-(3.29+0.01))/0.01
        tamth3=(10-(6.7+0.01))/0.01

        tramp21= np.arange(0, int(3.29)+1 ,1)
        tramp22= np.arange(int(3.29+0.01)+1, int(6.7)+1, 1)
        tramp23= np.arange(int(6.7+0.01)+1, 10+1, 1) 

        h1= (3.4/3.29)*(tramp21)
        h2= 3.4+(tramp22*0) 
        h3= (-3.4/3.29)*tramp23 + (34/3.29)

        ax2.set_title('Gráfica h[n]')
        
        th= np.concatenate( (tramp21,tramp22,tramp23), axis=None )
        h= np.concatenate( (h1,h2,h3), axis=None )
        ax2.stem(th,h)
        ax2.legend(['h[n]'],loc='upper left')

    st.pyplot(fig)

    if Convolucion:
        ConvDiscreto(tx,x,th,h)

if Fourier:    
    if n!=0:
        RepFourier(tx,x,th,h,tinicialx,tfinalx,n)
    else:
        st.write(""" 
        # El número de armonicos debe ser mayor que 0, preferiblemente mayor que 1.

        """)

