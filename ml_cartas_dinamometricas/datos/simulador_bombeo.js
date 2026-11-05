// Copia del núcleo de bombeo_mecanico/bombeo.html, usada para generar el dataset de entrenamiento.
// Núcleo físico del bombeo mecánico: cinemática del AIB, ecuación de onda en la sarta y bomba de fondo.
const g=9.81, BBL=0.158987;
function makeModel(o){
  const M=Object.assign({
    // Geometría del AIB (clase I convencional) [m]
    A:3.0, C:2.4, P:3.3, R:0.95, I:2.4, H:3.3,
    // Sarta de varillas (acero 7/8")
    L:1500, Ar:3.88e-4, E:2.07e11, rhoS:7850, damp:0.35, N:50,
    // Bomba y fluidos
    Dp:0.04445, rhoF:900, Ptub:0.8e6, Pcas:0.3e6, spacing:0.30, fric:500,
    // Reservorio (IPR lineal) y nivel en el anular (escala de tiempo acelerada)
    Pres:9e6, AOF:360, Aann:1.2e-5, hf0:250,
    spm:8, GVF:0, leakTV:0, leakSV:0, parted:false, partFrac:0.6, tag:false, Mcb:0
  },o||{});
  M.Ap=Math.PI*M.Dp*M.Dp/4; M.v=Math.sqrt(M.E/M.rhoS); M.geff=g*(1-M.rhoF/M.rhoS);
  M.Pdis=M.rhoF*g*M.L+M.Ptub;
  return M;
}
// Cinemática: ángulo de la viga β para un ángulo de manivela θ (resuelve la longitud de la biela)
function beamAngle(M,th){
  // Pivote de la viga B=(0,H), eje de manivela O=(I,0), extremo trasero E=(C cosβ, H + C sinβ)
  const cx=M.I+M.R*Math.cos(th), cy=M.R*Math.sin(th);
  let lo=-0.9, hi=0.9;
  const f=b=>{const ex=M.C*Math.cos(b), ey=M.H+M.C*Math.sin(b); return Math.hypot(ex-cx,ey-cy)-M.P;};
  // f es creciente en β cerca de la operación (E sube, se aleja del perno)
  for(let i=0;i<60;i++){const m=(lo+hi)/2; if(f(m)>0) hi=m; else lo=m;}
  return (lo+hi)/2;
}
function kinTable(M){
  const n=720, b=[], y=[];
  for(let i=0;i<=n;i++){ const th=i/n*2*Math.PI; const be=beamAngle(M,th); b.push(be); y.push(-M.A*be); }
  const ymin=Math.min(...y), ymax=Math.max(...y);
  const tf=y.map((_,i)=>{const a=y[(i+1)%n], c=y[(i-1+n)%n]; return (a-c)/(2*2*Math.PI/n);});
  return {n,b,y:y.map(v=>v-ymin),stroke:ymax-ymin,tf};
}
function kinAt(K,th){ const t=((th%(2*Math.PI))+2*Math.PI)%(2*Math.PI); const f=t/(2*Math.PI)*K.n; const i=Math.floor(f), w=f-i;
  const lerp=(arr)=>arr[i]*(1-w)+arr[Math.min(i+1,K.n)]*w; return {y:lerp(K.y),beta:lerp(K.b),tf:lerp(K.tf)}; }

function makeState(M){
  const K=kinTable(M), N=M.N, dx=M.L/N;
  const S={K,dx,u:new Float64Array(N+1),up:new Float64Array(N+1),th:0,t:0,hf:M.hf0,
    Vl:0,ng:0,yref:0,prevY:0,dir:1,produced:0,lastFill:1,fills:[],strokeLiq:0,
    Fp:0,Pch:0,PRL:0,TV:false,SV:false,cardS:[],cardD:[],lastCycle:{S:[],D:[]},cycle:0,qPump:0,qRes:0,torque:0,tq:[]};
  // Equilibrio estático en el fondo de la carrera con la bomba cargada
  const Fp0=(M.Pdis-(M.Pcas+M.rhoF*g*M.hf0))*M.Ap;
  for(let i=0;i<=N;i++){ const x=i*dx; // u(x)=∫ T/(EA) dx, T(x)=Fp0+ρA geff (L−x)
    S.u[i]=(Fp0*x+M.rhoS*M.Ar*M.geff*(M.L*x-x*x/2))/(M.E*M.Ar); S.up[i]=S.u[i]; }
  // Por la convención de signos, la superficie empieza en y=0 (fondo de carrera)
  S.th=Math.PI*1.5; const k0=kinAt(K,S.th); const off=k0.y; for(let i=0;i<=N;i++){S.u[i]-=off; S.up[i]-=off;}
  const yp=-S.u[N]; S.yref=yp-M.spacing; S.prevY=yp;
  const V=M.Ap*(yp-S.yref); S.Vl=V*0.998; S.ng=(M.Pcas+M.rhoF*g*M.hf0)*V*0.002;
  return S;
}
// Un paso de integración (dt en s)
function stepModel(M,S,dt){
  const N=M.N, dx=S.dx, v2=M.v*M.v, c=M.damp, w=M.spm*2*Math.PI/60;
  const ph0=((S.th-S.K.thBot)%(2*Math.PI)+2*Math.PI)%(2*Math.PI);
  S.th+=w*dt; S.t+=dt;
  const ph1=((S.th-S.K.thBot)%(2*Math.PI)+2*Math.PI)%(2*Math.PI);
  S.newCycle=ph1<ph0; S.atTop=false;
  if(S.newCycle){ S.cycle++; S.lastCycle={S:S.cardS,D:S.cardD,T:S.tq,prod:S.produced-S.prodAtCycle,dur:S.t-S.tAtCycle}; S.cardS=[]; S.cardD=[]; S.tq=[]; S.prodAtCycle=S.produced; S.tAtCycle=S.t; }
  if(ph0<S.K.phTop&&ph1>=S.K.phTop){ S.atTop=true; }
  const k=kinAt(S.K,S.th); const ypr=k.y;
  const Nb=M.parted?Math.round(N*M.partFrac):N; // nodo inferior efectivo
  // --- Bomba (con el estado previo de la sarta) ---
  const Pin=M.Pcas+M.rhoF*g*Math.max(0,S.hf);
  let yp=-S.u[N], Fp=0;
  if(!M.parted){
    const dy=yp-S.prevY; S.prevY=yp;
    const V=M.Ap*Math.max(0.005,yp-S.yref);
    // fugas
    let P=S.ng/Math.max(1e-9,V-S.Vl);
    if(M.leakTV>0&&P<M.Pdis){ const back=1.5e-11*M.leakTV*(M.Pdis-Math.max(P,Pin))*dt; S.Vl+=back; S.produced-=back; } // lo que se fuga desde la tubería se resta de lo producido
    if(M.leakSV>0&&P>Pin){ const out=Math.min(S.Vl,1.5e-11*M.leakSV*(Math.min(P,M.Pdis)-Pin)*dt); S.Vl-=out; S.hf+=out/M.Aann; }
    P=S.ng/Math.max(1e-9,V-S.Vl);
    S.TV=false; S.SV=false;
    if(P<Pin){ // válvula fija abre: entra mezcla desde el anular
      S.SV=true; const space=(V-S.Vl)-S.ng/Pin;
      if(space>0){ // Con nivel sobre la bomba entra líquido sin restricción; si el pozo está "bombeado" (nivel ≈ 0) solo entra lo que aporta el reservorio
        const annLiq=S.hf>20?Infinity:Math.max(0,S.hf)*M.Aann; let Lin=Math.min(space*(1-M.GVF),annLiq);
        S.Vl+=Lin; S.hf=Math.max(0,S.hf-Lin/M.Aann); S.qPump+=Lin; S.ng+=(space-Lin)*Pin; }
      P=Pin;
    } else if(P>M.Pdis){ // válvula viajera abre: descarga a la tubería
      S.TV=true; const needG=S.ng/M.Pdis; let out=S.Vl-(V-needG);
      if(out>0){ out=Math.min(out,S.Vl); S.Vl-=out; S.produced+=out; }
      if(V-S.Vl<needG){ S.ng=M.Pdis*(V-S.Vl); }
      P=M.Pdis;
    }
    S.ng=Math.max(S.ng,1e-7*V*Pin+0.0004*V*M.Pcas); // siempre un poco de gas disuelto liberado
    S.Pch=P;
    if(S.atTop){ const V2=M.Ap*Math.max(0.02,yp-S.yref); S.lastFill=Math.min(1,S.Vl/V2); S.fills.push(S.lastFill); if(S.fills.length>6) S.fills.shift(); }
    Fp=(M.Pdis-P)*M.Ap+M.fric*(dy>=0?1:-1);
    if(M.tag&&yp<S.yref+0.02){ Fp-=Math.min(40000,6e5*(S.yref+0.02-yp)); } // la bomba golpea abajo
  }
  // --- Ecuación de onda amortiguada (diferencias finitas explícitas) ---
  const un=new Float64Array(N+1); const r=v2*dt*dt/(dx*dx), a1=1+c*dt/2, a2=1-c*dt/2;
  un[0]=-ypr;
  for(let i=1;i<=Nb;i++){
    let uR;
    if(i===Nb){ uR=S.u[i-1]+2*dx*(Fp/(M.E*M.Ar)); }
    else uR=S.u[i+1];
    un[i]=(2*S.u[i]-a2*S.up[i]+r*(uR-2*S.u[i]+S.u[i-1])+dt*dt*M.geff)/a1;
  }
  if(Nb<N){ for(let i=Nb+1;i<=N;i++){ un[i]=S.u[i]+(S.u[i]-S.up[i])*0.98+0.5*g*dt*dt*0; } }
  S.up=S.u; S.u=un;
  // Cargas
  S.PRL=M.E*M.Ar*(-3*S.u[0]+4*S.u[1]-S.u[2])/(2*dx);
  S.Fp=Fp;
  // Reservorio → anular
  const qres=M.AOF*BBL/86400*Math.max(0,1-Pin/M.Pres); S.qRes=qres; S.hf+=qres*dt/M.Aann;
  // Par en el reductor
  S.torque=k.tf*S.PRL - M.Mcb*Math.sin(S.th-S.K.thref);
  return {ypr,yp:-S.u[N],tf:k.tf};
}
// Asentamiento: dos ciclos para conocer la carrera real del pistón y espaciar la bomba (como en el campo)
function settle(M,S){
  initPhase(S); const T=60/M.spm, dt=0.001; let ymin=1e9;
  const save={tag:M.tag}; M.tag=false; S.yref=-1e6; // cámara grande durante el asentamiento
  const hf=S.hf;
  while(S.t<2*T){ stepModel(M,S,dt); S.hf=hf; if(S.t>T) ymin=Math.min(ymin,-S.u[M.N]); }
  S.yref=ymin-M.spacing; const V=M.Ap*M.spacing; const Pin=M.Pcas+M.rhoF*g*S.hf;
  S.Vl=V*0.998; S.ng=Pin*V*0.002; S.produced=0; S.qPump=0; S.t=0; S.cycle=0; S.prodAtCycle=0; S.tAtCycle=0; S.cardS=[]; S.cardD=[]; S.fills=[]; S.lastFill=1;
  M.tag=save.tag;
}
function initPhase(S){ // fase del contrapeso alineada con el factor de torque máximo; ángulos de fondo y tope de carrera
  let ib=0, it=0; for(let i=0;i<S.K.n;i++){ if(S.K.y[i]<S.K.y[ib]) ib=i; if(S.K.y[i]>S.K.y[it]) it=i; }
  S.K.thBot=ib/S.K.n*2*Math.PI; S.K.phTop=((it-ib+S.K.n)%S.K.n)/S.K.n*2*Math.PI; S.prodAtCycle=0; S.tAtCycle=0;
  let im=0; for(let i=0;i<S.K.n;i++) if(S.K.tf[i]>S.K.tf[im]) im=i; S.K.thref=im/S.K.n*2*Math.PI-Math.PI/2;
}
if(typeof module!=='undefined') module.exports={makeModel,makeState,stepModel,kinTable,kinAt,initPhase,settle,BBL,g};
