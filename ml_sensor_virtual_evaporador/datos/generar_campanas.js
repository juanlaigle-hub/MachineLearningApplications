// Genera campañas de operación del evaporador de 5 efectos, como las que guardaría el historiador de planta (PI, SCADA).
// Uso: node generar_campanas.js [n_campanas] [horas_por_campana] [semilla] [n_ensayos]
// Además de la operación normal (control de °Brix en automático) se generan campañas de ENSAYO:
// alimentación en manual con escalones aleatorios, como las pruebas de identificación que se hacen en planta
// para que los datos tengan información sobre la dinámica (en lazo cerrado el °Brix casi no varía).
// Salida:
//   historico.csv     → una fila por minuto con los sensores de proceso (con ruido) y los valores reales (solo para evaluar)
//   laboratorio.csv   → análisis de laboratorio de HMF y color cada 2 h (con error de laboratorio), como en una juguera real
const fs=require('fs'), path=require('path');
const E=require('./simulador_evaporador.js');
const NC=+(process.argv[2]||12), HRS=+(process.argv[3]||24), SEED=+(process.argv[4]||11), NENS=+(process.argv[5]||2);
let s=SEED>>>0; const rnd=()=>{ s+=0x6D2B79F5; let t=s; t=Math.imul(t^t>>>15,t|1); t^=t+Math.imul(t^t>>>7,t|61); return ((t^t>>>14)>>>0)/4294967296; };
const U=(a,b)=>a+(b-a)*rnd();
const gauss=()=>{let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
// Ruido de instrumentos (desvío estándar)
const RUIDO={T:0.2, P:0.3, F:0.01, steam:0.02, Bf:0.1, Tcw:0.2, Bmeas:0.15};
const LAB_ERR=0.05; // 5 % de error relativo del laboratorio

const H=[], LAB=[];
const hdr=['campana','tipo','t_min','Ts','T1','T2','T3','T4','T5','P1','P2','P3','P4','P5','F_th','vapor_th','Bf','Tcw','Bsp','enfriador','horas_tanque','Brix_refract',
           'real_Brix','real_HMF','real_color','real_foul'];
for(let c=1;c<=NC+NENS;c++){
  const ensayo=c>NC; const HRSc=ensayo?8:HRS;
  const P=E.makePlant({}); const S=E.makeState(P); S.Fbias=P.F;
  // Planificación de la campaña: arranca recién lavada (CIP) y se ensucia de a poco
  const foulEnd=U(0.25,0.45), Tcw0=U(22,30), fase=U(0,2*Math.PI);
  let nextTs=0, nextBf=0, nextTank=U(3,10)*3600, tankOffUntil=-1;
  P.tankHours=U(1,4); P.Bsp=70; const spLow=rnd()<0.35?{t0:U(4,Math.max(5,HRS-6))*3600,dur:U(2,4)*3600,v:U(65,68)}:null;
  for(let i=0;i<2400;i++) E.step(P,S,0.5); S.t=0; // estado inicial estacionario
  const total=HRSc*3600, dt=0.5; let nextF=ensayo?1800:Infinity;
  if(ensayo){ P.auto=false; P.F=S.F; }
  for(let t=0;t<=total;t+=dt){
    // en los ensayos el vapor se mantiene moderado para que los escalones no activen el enclavamiento de °Brix alto
    if(t>=nextTs){ P.Ts=ensayo?U(102,106):U(100,114); nextTs=t+U(1.5,4)*3600; }
    if(t>=nextBf){ P.Bf=U(10.2,13.5); nextBf=t+U(2,6)*3600; }
    P.Tcw=Tcw0+4*Math.sin(2*Math.PI*t/86400+fase);
    P.foul=foulEnd*t/total;
    if(t>=nextTank){ tankOffUntil=t+U(1,3)*3600; nextTank=t+U(6,14)*3600; }
    P.tankCool=t>tankOffUntil;
    P.Bsp=spLow&&t>=spLow.t0&&t<spLow.t0+spLow.dur?spLow.v:70;
    if(ensayo&&t>=nextF){ P.F=Math.min(25,Math.max(15,P.F*3.6+U(-4,4)))/3.6; nextF=t+U(20,40)*60; }
    E.step(P,S,dt);
    if(t%60===0){
      const r=v=>v.toFixed(3);
      H.push([c,ensayo?'ensayo':'operacion',(t/60).toFixed(0),r(P.Ts+RUIDO.T*gauss()),...S.T.map(v=>r(v+RUIDO.T*gauss())),...S.Pk.map(v=>r(v+RUIDO.P*gauss())),
        r(S.F*3.6*(1+RUIDO.F*gauss())),r(S.steam*3.6*(1+RUIDO.steam*gauss())),r(P.Bf+RUIDO.Bf*gauss()),r(P.Tcw+RUIDO.Tcw*gauss()),r(P.Bsp),P.tankCool?1:0,r(P.tankHours),
        r(S.Bmeas+RUIDO.Bmeas*gauss()),r(S.x[4]*100),r(S.hProd*0.112),(S.cProd*0.112).toFixed(5),r(P.foul)].join(','));
      if(t>0&&t%7200===0) LAB.push([c,(t/60).toFixed(0),(S.hProd*0.112*(1+LAB_ERR*gauss())).toFixed(3),(S.cProd*0.112*(1+LAB_ERR*gauss())).toFixed(5)].join(','));
    }
  }
  process.stderr.write('campaña '+c+' lista\n');
}
fs.writeFileSync(path.join(__dirname,'historico.csv'),hdr.join(',')+'\n'+H.join('\n')+'\n');
fs.writeFileSync(path.join(__dirname,'laboratorio.csv'),'campana,t_min,HMF_lab,color_lab\n'+LAB.join('\n')+'\n');
process.stderr.write('Listo: '+H.length+' filas de historiador, '+LAB.length+' análisis de laboratorio\n');
