// Genera un dataset etiquetado de cartas dinamométricas con el simulador físico de bombeo.
// Uso: node generar_dataset.js [muestras_por_clase] [semilla] [modo]
//   modo "base"   → pozos dentro del rango de entrenamiento (dataset.csv)
//   modo "fuera"  → pozos fuera de rango para probar la generalización (dataset_fuera_de_rango.csv)
//   modo "barrido"→ un pozo con el aporte o la fuga variando de a poco, para verificar la fenomenología (dataset_barrido.csv)
// Cada fila: metadatos del pozo + carta de fondo y de superficie remuestreadas (32 + 32 puntos cada una).
const fs=require('fs'), path=require('path');
const C=require('./simulador_bombeo.js');
const PER=+(process.argv[2]||400), SEED=+(process.argv[3]||7), MODE=process.argv[4]||'base';

// Generador pseudoaleatorio reproducible (mulberry32)
let s=SEED>>>0; const rnd=()=>{ s+=0x6D2B79F5; let t=s; t=Math.imul(t^t>>>15,t|1); t^=t+Math.imul(t^t>>>7,t|61); return ((t^t>>>14)>>>0)/4294967296; };
const U=(a,b)=>a+(b-a)*rnd(), pick=a=>a[Math.floor(rnd()*a.length)];
const gauss=()=>{let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};

const CLASES=['normal','golpe_de_fluido','interferencia_de_gas','fuga_valvula_viajera','fuga_valvula_fija','varilla_cortada','golpe_abajo'];

function pozoAleatorio(){
  if(MODE==='fuera') return {L:U(2250,2600), spm:U(11,12.5), Dp:pick([1.06,2.5])*0.0254, rhoF:U(1000,1050), Ptub:U(1.6,2.0)*1e6, damp:U(0.2,0.25), fric:U(900,1200)};
  return {L:U(1000,2200), spm:U(5,11), Dp:pick([1.25,1.5,1.75,2.0,2.25])*0.0254, rhoF:U(850,1000), Ptub:U(0.4,1.5)*1e6, damp:U(0.25,0.5), fric:U(300,900)};
}
// Simula un pozo y devuelve el último ciclo completo (cartas de superficie y fondo) y el llenado
function simular(base,falla){
  const M=C.makeModel(Object.assign({},base,{Mcb:60000})); const S=C.makeState(M);
  // capacidad teórica de la bomba para fijar el aporte del reservorio
  const Qp=M.Ap*S.K.stroke*base.spm*1440/C.BBL; // bbl/d
  M.AOF=Qp*2.2/(1-M.Pcas/M.Pres); M.hf0=U(150,600); S.hf=M.hf0;
  if(falla.tipo==='golpe_de_fluido'){ M.AOF=falla.frac*Qp/(1-M.Pcas/M.Pres); }
  if(falla.tipo==='interferencia_de_gas'){ M.GVF=falla.gvf; }
  if(falla.tipo==='fuga_valvula_viajera'){ M.leakTV=falla.sev; }
  if(falla.tipo==='fuga_valvula_fija'){ M.leakSV=falla.sev; }
  C.settle(M,S);
  if(falla.tipo==='golpe_de_fluido'&&!falla.nivelLibre){ S.hf=0; }
  if(falla.nivelLibre){ S.hf=40; } // el nivel arranca bajo y evoluciona solo según el aporte
  if(falla.tipo==='golpe_abajo'){ M.tag=true; S.yref+=M.spacing-0.03; }
  if(falla.tipo==='varilla_cortada'){ M.parted=true; M.partFrac=falla.frac; }
  const T=60/M.spm, dt=0.001; let cS=[], cD=[], last=null, cyc=S.cycle, prod0=0;
  const NCIC=falla.ciclos||4;
  while(S.t<NCIC*T+0.5){
    C.stepModel(M,S,dt);
    const k=C.kinAt(S.K,S.th);
    if(S.cycle!==cyc){ cyc=S.cycle; if(S.cycle>=NCIC-1) last={S:cS,D:cD,fill:S.lastFill,prod:S.produced-prod0,dur:T}; cS=[]; cD=[]; prod0=S.produced; }
    cS.push([k.y,S.PRL]); cD.push([-S.u[M.N]-S.yref-M.spacing,S.Fp]);
  }
  const Pin=M.Pcas+M.rhoF*C.g*Math.max(0,S.hf); const Fo=(M.Pdis-Pin)*M.Ap;
  return {last,Fo,stroke:S.K.stroke,Qp,fill:S.lastFill};
}
// Ruido de dinamómetro: ±0,3 kN en carga y ±1 cm en posición
const ruido=card=>card.map(([x,f])=>[x+0.01*gauss(),f+300*gauss()]);
// Remuestreo: separa subida y bajada en los extremos de posición e interpola N puntos equiespaciados en cada una
function remuestrear(card,N=32){
  const xs=card.map(p=>p[0]); let iMin=0,iMax=0; xs.forEach((x,i)=>{ if(x<xs[iMin]) iMin=i; if(x>xs[iMax]) iMax=i; });
  const x0=xs[iMin], x1=xs[iMax], span=Math.max(1e-6,x1-x0);
  const seg=(a,b)=>{ const out=[]; let i=a; const n=card.length; const pts=[]; while(true){ pts.push(card[i]); if(i===b) break; i=(i+1)%n; } return pts; };
  const up=seg(iMin,iMax), down=seg(iMax,iMin);
  const interp=(pts,asc)=>{ const r=[]; for(let k=0;k<N;k++){ const xt=x0+span*(asc?k/(N-1):1-k/(N-1)); let best=pts[0], bd=1e9;
      // primer cruce en el sentido de avance (más robusto que el más cercano con oscilaciones)
      for(let j=1;j<pts.length;j++){ const [xa,fa]=pts[j-1],[xb,fb]=pts[j]; if((xa-xt)*(xb-xt)<=0&&xa!==xb){ const w=(xt-xa)/(xb-xa); best=[xt,fa+w*(fb-fa)]; bd=0; break; } const d=Math.abs(xb-xt); if(d<bd){bd=d; best=pts[j];} }
      r.push(best[1]); } return r; };
  return {up:interp(up,true),down:interp(down,false),span};
}
function filaDe(res,base,falla){
  const Sn=ruido(res.last.S), Dn=ruido(res.last.D); const S=remuestrear(Sn), D=remuestrear(Dn);
  const k=v=>(v/1000).toFixed(3);
  return {meta:{clase:falla.tipo,L:base.L.toFixed(0),spm:base.spm.toFixed(2),Dp_in:(base.Dp/0.0254).toFixed(2),rhoF:base.rhoF.toFixed(0),Ptub_MPa:(base.Ptub/1e6).toFixed(2),
      severidad:(falla.frac??falla.gvf??falla.sev??0).toFixed(3),llenado:res.fill.toFixed(3),Fo_kN:k(res.Fo),carrera_sup_m:res.stroke.toFixed(3),carrera_bomba_m:D.span.toFixed(3),
      produccion_bpd:(res.last.prod/res.last.dur*86400/C.BBL).toFixed(1),
      // extremos sin remuestrear: el pico de compresión al tocar fondo dura muy poco y se pierde con 32 puntos
      fondo_min_kN:k(Math.min(...Dn.map(p=>p[1]))),fondo_max_kN:k(Math.max(...Dn.map(p=>p[1]))),sup_min_kN:k(Math.min(...Sn.map(p=>p[1]))),sup_max_kN:k(Math.max(...Sn.map(p=>p[1])))},
    sup:[...S.up,...S.down].map(k), fondo:[...D.up,...D.down].map(k)};
}
// Chequeos de fenomenología: se descartan muestras cuya física no corresponde a la etiqueta
function fenomenologiaOK(clase,res,ref){
  const f=res.fill, q=res.last.prod/res.last.dur, qref=ref?ref.last.prod/ref.last.dur:null;
  switch(clase){
    case 'normal': return f>0.95;
    case 'golpe_de_fluido': return f<0.88;
    case 'interferencia_de_gas': return f<0.92;
    case 'fuga_valvula_viajera': case 'fuga_valvula_fija': return qref>0&&q<0.92*qref; // pierde al menos 8 % de producción
    case 'varilla_cortada': return q<0.02*res.Qp*C.BBL/86400+1e-9;
    case 'golpe_abajo': { const fmin=Math.min(...res.last.D.map(p=>p[1])); return fmin<-3000; } // carga de compresión al tocar fondo
  }
  return true;
}
// Modo "barrido": un mismo pozo, una variable física que cambia de a poco (para verificar la fenomenología del modelo)
if(MODE==='barrido'){
  const base={L:1500,spm:8,Dp:1.75*0.0254,rhoF:900,Ptub:0.8e6,damp:0.35,fric:500};
  const filas=[];
  for(let k=0;k<=30;k++){ const frac=1.15-0.85*k/30; const res=simular(base,{tipo:'golpe_de_fluido',frac,nivelLibre:true,ciclos:10}); const f=filaDe(res,base,{tipo:'golpe_de_fluido',frac}); f.meta.barrido='aporte'; f.meta.param=frac.toFixed(3); filas.push(f); }
  for(let k=0;k<=20;k++){ const sev=5*k/20; for(const tipo of ['fuga_valvula_viajera','fuga_valvula_fija']){ const fz={tipo,sev}; const res=simular(base,fz); const f=filaDe(res,base,fz); f.meta.barrido=tipo; f.meta.param=sev.toFixed(3); filas.push(f);} }
  const mc=Object.keys(filas[0].meta); const hdr=[...mc,...[...Array(64)].map((_,i)=>'fondo_'+i),...[...Array(64)].map((_,i)=>'sup_'+i)];
  fs.writeFileSync(path.join(__dirname,'dataset_barrido.csv'),[hdr.join(','),...filas.map(r=>[...mc.map(c=>r.meta[c]),...r.fondo,...r.sup].join(','))].join('\n')+'\n');
  process.stderr.write('Barrido listo: '+filas.length+' cartas\n'); process.exit(0);
}
const out=[]; const stats={}; const ejemplos={};
const t0=Date.now();
for(const clase of CLASES){
  let ok=0, desc=0;
  let intentos=0;
  while(ok<PER){ if(++intentos>PER*6) throw new Error('Demasiados descartes en '+clase+': revisar la física o los rangos');
    const base=pozoAleatorio();
    const falla={tipo:clase};
    if(clase==='golpe_de_fluido') falla.frac=U(0.3,0.85);
    if(clase==='interferencia_de_gas') falla.gvf=U(0.15,0.45);
    if(clase.startsWith('fuga')) falla.sev=U(1.5,5);
    if(clase==='varilla_cortada') falla.frac=U(0.2,0.85);
    const res=simular(base,falla); if(!res.last){ desc++; continue; }
    const ref=clase.startsWith('fuga')?simular(base,{tipo:'normal'}):null;
    if(!fenomenologiaOK(clase,res,ref)){ desc++; continue; }
    const fila=filaDe(res,base,falla); out.push(fila); ok++;
    if(!ejemplos[clase]) ejemplos[clase]={sup:res.last.S.filter((_,i)=>i%10===0),fondo:res.last.D.filter((_,i)=>i%10===0),meta:fila.meta};
  }
  stats[clase]={aceptadas:ok,descartadas:desc}; process.stderr.write(clase+': '+ok+' aceptadas, '+desc+' descartadas por fenomenología ('+((Date.now()-t0)/1000).toFixed(0)+' s)\n');
}
const metaCols=Object.keys(out[0].meta);
const hdr=[...metaCols,...[...Array(64)].map((_,i)=>'fondo_'+i),...[...Array(64)].map((_,i)=>'sup_'+i)];
const lines=[hdr.join(',')]; for(const r of out) lines.push([...metaCols.map(c=>r.meta[c]),...r.fondo,...r.sup].join(','));
const name=MODE==='fuera'?'dataset_fuera_de_rango.csv':'dataset.csv';
fs.writeFileSync(path.join(__dirname,name),lines.join('\n')+'\n');
if(MODE==='base') fs.writeFileSync(path.join(__dirname,'ejemplos_por_clase.json'),JSON.stringify(ejemplos));
fs.writeFileSync(path.join(__dirname,'resumen_'+MODE+'.json'),JSON.stringify({seed:SEED,porClase:PER,stats},null,1));
process.stderr.write('Listo: '+out.length+' muestras → '+name+'\n');
