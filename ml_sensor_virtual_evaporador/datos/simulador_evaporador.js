// Copia del núcleo de evaporador_jugo/evaporador.html, usada para generar los datos de entrenamiento.
// Evaporador de 5 efectos de película descendente para jugo de manzana: núcleo físico.
const NE=5, LAMBDA=2330, CP=3.8, X_MAX=0.80; // kJ/kg, kJ/(kg·K) (cp medio del jugo)
function tsat(P){ // °C, P en kPa (Antoine para agua, 1–200 kPa)
  const mmHg=P*7.50062; return 1730.63/(8.07131-Math.log10(mmHg))-233.426; }
function psat(T){ return Math.pow(10,8.07131-1730.63/(233.426+T))/7.50062; } // kPa
function bpe(B){ return 2.5*B/(100-B); } // elevación del punto de ebullición [K], B en °Brix
function makePlant(o){
  const P=Object.assign({
    Ts:105, Tcw:28, Bf:12, Tf:60, F:20000/3600, // vapor vivo °C, agua de condensador °C, °Brix y T alimentación, kg/s
    UA:[230,220,200,170,120], UAc:300, foul:0, hold:[300,250,200,150,120], // kW/K, kg
    Bsp:70, auto:true, Kp:0.07, Ti:150, Fmin:10000/3600, // caudal mínimo de mojado de los tubos de película descendente
    Fmax:28000/3600,
    tankCool:true, tankT:10, tankHours:2,
    // cinética de daño térmico (por kg de sólidos solubles)
    kH:0.18, EaH:130e3, kC:0.0011, EaC:100e3, Tref:90,
    aroma:9
  },o||{});
  return P;
}
function makeState(P){
  const x=[0.14,0.18,0.25,0.38,0.70];
  return {x:x.slice(), h:[0,0,0,0,0], c:[0,0,0,0,0], T:[95,85,72,60,48], V:[0,0,0,0,0], L:[0,0,0,0,0], Q:[0,0,0,0,0], Pk:[0,0,0,0,0],
    t:0, I:0, Bmeas:70, F:P.F, Fbias:P.F, hProd:0, cProd:0, steam:0, Qc:0, Pc:0, aroma:0};
}
const Rg=8.314;
function rate(k0,Ea,T,x,Tref){ return k0*Math.exp(-Ea/Rg*(1/(T+273.15)-1/(Tref+273.15)))*Math.exp(3.0*(x-0.12)); }
// Reparte el calor: con Q1 dado se recorre la cadena vapor → efecto 1 → … → efecto 5 → condensador.
// Q1 se ajusta por bisección hasta que el condensador absorba exactamente el vapor del 5º efecto.
function chain(P,S,Q1,UA,be){
  const F=S.F; let Tc=P.Ts, Lprev=F, Tprev=P.Tf; const T=[],Tv=[],V=[],Q=[],L=[];
  for(let i=0;i<NE;i++){
    Q[i]=i===0?Q1:V[i-1]*LAMBDA;
    T[i]=Tc-Q[i]/UA[i];
    const sens=Lprev*CP*(Tprev-T[i]);
    V[i]=Math.max(0,Math.min(Lprev*0.97,(Q[i]+sens)/LAMBDA));
    L[i]=Lprev-V[i]; Tv[i]=T[i]-be[i]; Tc=Tv[i]; Tprev=T[i]; Lprev=L[i];
  }
  const res=V[NE-1]*LAMBDA-P.UAc*(Tv[NE-1]-P.Tcw);
  return {T,Tv,V,Q,L,res};
}
function energy(P,S){
  const foulK=[0.6,0.7,0.8,0.9,1.0]; // el ensuciamiento pesa más en los efectos concentrados
  // U cae fuerte con la viscosidad cerca de 80 °Brix: la evaporación se autolimita de forma suave (sin topes duros)
  const visc=x=>1/(1+Math.exp((x-X_MAX-0.03)/0.03));
  const UA=P.UA.map((u,i)=>u*(1-P.foul*foulK[i])*visc(S.x[i]));
  const be=S.x.map(v=>bpe(v*100));
  let lo=0, hi=UA[0]*(P.Ts-P.Tcw), r=null;
  for(let it=0;it<60;it++){ const m=(lo+hi)/2; r=chain(P,S,m,UA,be); if(r.res>0) hi=m; else lo=m; }
  S.Q=r.Q; S.V=r.V; S.L=r.L; S.T=r.T; S.Tv=r.Tv; S.Pk=r.Tv.map(psat); S.UAe=UA; S.be=be;
  S.steam=r.Q[0]/2240; S.Qc=r.V[NE-1]*LAMBDA; S.Pc=psat(r.Tv[NE-1]);
}
// Paso dinámico: sólidos, HMF y color por efecto (base sólidos), control de °Brix, tanque de producto
function step(P,S,dt){
  energy(P,S);
  const F=S.F; let Lin=F, xin=P.Bf/100, hin=0, cin=0;
  for(let i=0;i<NE;i++){
    const M=P.hold[i], L=S.L[i], V=S.V[i], x=S.x[i];
    const dx=(Lin*(xin-x)+V*x)/M;
    // HMF y color por kg de sólidos: llegan con los sólidos y se forman según T y °Brix
    const rh=rate(P.kH,P.EaH,S.T[i],x,P.Tref), rc=rate(P.kC,P.EaC,S.T[i],x,P.Tref);
    const sol=M*x, solIn=Lin*xin;
    const dh=(solIn*(hin-S.h[i]))/sol+rh, dc=(solIn*(cin-S.c[i]))/sol+rc;
    S.x[i]=Math.min(0.85,Math.max(0.05,x+dx*dt)); S.h[i]+=dh*dt; S.c[i]+=dc*dt;
    Lin=L; xin=S.x[i]; hin=S.h[i]; cin=S.c[i];
  }
  S.t+=dt;
  // Refractómetro con retardo de 20 s
  S.Bmeas+=(S.x[NE-1]*100-S.Bmeas)*dt/20;
  // Control PI del °Brix final con el caudal de alimentación
  if(P.auto){ const e=S.Bmeas-P.Bsp; // si el °Brix está alto, se puede alimentar más
    // PI en forma posicional con integral acotada al rango del actuador (anti-windup)
    S.I=Math.max(P.Fmin-S.Fbias,Math.min(P.Fmax-S.Fbias,S.I+P.Kp/P.Ti*e*dt));
    const Fc=S.Fbias+P.Kp*e+S.I;
    S.F=Math.min(P.Fmax,Math.max(P.Fmin,Fc)); } else { S.F=P.F; }
  // Enclavamiento de °Brix alto (independiente del control, activo también en manual): si el concentrado
  // supera 76 °Brix se fuerza más alimentación, como en las plantas reales, para no quemar el producto.
  S.ovr=S.ovr||0; const B5=S.x[NE-1]*100;
  S.ovr=B5>76?Math.min(P.Fmax,S.ovr+0.03*(B5-76)*dt):Math.max(0,S.ovr-0.002*dt);
  if(S.ovr>0) S.F=Math.min(P.Fmax,S.F+S.ovr);
  // Producto: tiempo en tanque (enfriado o caliente)
  const Tt=P.tankCool?P.tankT:S.T[NE-1];
  const tt=P.tankHours*3600;
  S.hProd=S.h[NE-1]+rate(P.kH,P.EaH,Tt,S.x[NE-1],P.Tref)*tt;
  S.cProd=S.c[NE-1]+rate(P.kC,P.EaC,Tt,S.x[NE-1],P.Tref)*tt;
  // Recuperación de aromas: fracción despojada en el 1er efecto
  S.aroma=1-Math.exp(-P.aroma*S.V[0]/F);
}
if(typeof module!=='undefined') module.exports={makePlant,makeState,energy,step,tsat,psat,bpe,rate,NE,LAMBDA,CP};
