"""Gráfica animada de la simulación Monte Carlo (HTML + canvas, se muestra con components.html)."""
from __future__ import annotations

import json

PLANTILLA = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html,body{margin:0;background:transparent;color:#F4EDE0;font-family:Inter,sans-serif;overflow:hidden}
.marco{position:relative;border:1px solid rgba(244,237,224,.16);border-radius:22px;overflow:hidden;
  background:repeating-linear-gradient(90deg,rgba(244,237,224,.03) 0 2px,transparent 2px 10px),
  radial-gradient(500px 300px at 75% 30%,rgba(255,107,91,.16),transparent 70%),
  radial-gradient(500px 300px at 15% 80%,rgba(125,155,209,.14),transparent 70%),rgba(14,24,48,.72)}
canvas{display:block;width:100%;height:__ALTO__px}
.top{position:absolute;left:18px;right:18px;top:14px;display:flex;justify-content:space-between;align-items:center;pointer-events:none}
.tit{font:500 15px 'Inter Tight',sans-serif;letter-spacing:-.01em}
.tit b{color:#FF6B5B;font-weight:600}
.cont{font-size:12px;color:#9AA3B8;margin-left:10px}
button{pointer-events:auto;cursor:pointer;background:rgba(244,237,224,.06);color:#F4EDE0;border:1px solid rgba(244,237,224,.3);
  border-radius:999px;padding:7px 14px;font:500 13px Inter,sans-serif;transition:background .2s}
button:hover{background:rgba(255,107,91,.25);border-color:#FF6B5B}
.leyenda{position:absolute;left:74px;bottom:12px;display:flex;gap:16px;font-size:12px;color:#D9D2C4;pointer-events:none;flex-wrap:wrap}
.leyenda i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
</style></head><body><div class="marco">
<div class="top"><div><span class="tit">__TITULO__</span><span class="cont" id="cont"></span></div><button id="rep">↻ Volver a simular</button></div>
<canvas id="c"></canvas>
<div class="leyenda"><span><i style="background:#FF6B5B"></i>Escenarios que terminan arriba</span><span><i style="background:#7D9BD1"></i>Terminan abajo</span>
<span><i style="background:rgba(255,107,91,.35);height:8px"></i>90% de los escenarios</span><span><i style="background:#F4EDE0"></i>Mediana</span></div>
</div>
<script>
const D = __DATOS__;
const N_TOTAL = __NTOTAL__, MON = "__MONEDA__", ALTO = __ALTO__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const CORAL='#FF6B5B', ACERO='#7D9BD1', CREMA='#F4EDE0';
let W=0,H=0,dpr=1,t0=performance.now(),hover=null,anim=true;
const M={l:70,r:168,t:74,b:66};
function fmt(v){const a=Math.abs(v);const d=a>=1000?0:a>=100?1:a>=1?2:5;
  return v.toLocaleString('es-CO',{minimumFractionDigits:d,maximumFractionDigits:d});}
function etiquetaX(f){if(f===0)return 'Hoy';const d=Math.round(D.dias*f);
  if(D.dias<=95)return '+'+d+' días';if(d>=365&&Math.abs(d/365-Math.round(d/365))<.05)return '+'+Math.round(d/365)+(Math.round(d/365)>1?' años':' año');
  return '+'+Math.round(d/30.4)+' meses';}
const ease=x=>x<=0?0:x>=1?1:1-Math.pow(1-x,3);
const easeIO=x=>x<=0?0:x>=1?1:x<.5?4*x*x*x:1-Math.pow(-2*x+2,3)/2;
let ymin,ymax;
(function(){const v=[D.s0,...D.bandas['5'],...D.bandas['95'],D.bordes[0],D.bordes[D.bordes.length-1]];
  if(D.objetivo)v.push(D.objetivo);ymin=Math.min(...v);ymax=Math.max(...v);const p=(ymax-ymin)*.08;ymin=Math.max(0,ymin-p);ymax+=p;})();
const X=f=>M.l+f*(W-M.l-M.r), Y=v=>M.t+(1-(v-ymin)/(ymax-ymin))*(H-M.t-M.b);
function tam(){dpr=window.devicePixelRatio||1;W=cv.clientWidth;H=ALTO;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}
function ejes(){ctx.save();ctx.font='12px Inter, sans-serif';ctx.fillStyle='#9AA3B8';ctx.strokeStyle='rgba(244,237,224,.08)';ctx.lineWidth=1;
  const n=5;for(let i=0;i<=n;i++){const v=ymin+(ymax-ymin)*i/n,y=Y(v);ctx.beginPath();ctx.setLineDash([2,6]);ctx.moveTo(M.l,y);ctx.lineTo(W-M.r,y);ctx.stroke();
    ctx.setLineDash([]);ctx.textAlign='right';ctx.fillText(fmt(v),M.l-10,y+4);}
  ctx.textAlign='center';[0,.25,.5,.75,1].forEach(f=>ctx.fillText(etiquetaX(f),X(f),H-M.b+22));ctx.restore();}
function lineaH(v,color,txt,alpha){const y=Y(v);ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.setLineDash([5,5]);ctx.lineWidth=1.2;
  ctx.beginPath();ctx.moveTo(M.l,y);ctx.lineTo(W-M.r,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=color;ctx.font='600 11px Inter, sans-serif';ctx.textAlign='left';
  ctx.fillText(txt,M.l+8,y-6);ctx.restore();}
function camino(c,prog){const n=D.x.length,k=prog*(n-1),e=Math.floor(k);ctx.beginPath();ctx.moveTo(X(D.x[0]),Y(c[0]));
  for(let j=1;j<=e;j++)ctx.lineTo(X(D.x[j]),Y(c[j]));
  if(e<n-1){const fr=k-e,x=D.x[e]+(D.x[e+1]-D.x[e])*fr,v=c[e]+(c[e+1]-c[e])*fr;ctx.lineTo(X(x),Y(v));return [X(x),Y(v)];}
  return [X(D.x[n-1]),Y(c[n-1])];}
function banda(a,b,color){ctx.beginPath();const A=D.bandas[a],B=D.bandas[b];
  D.x.forEach((x,i)=>i?ctx.lineTo(X(x),Y(A[i])):ctx.moveTo(X(x),Y(A[i])));
  for(let i=D.x.length-1;i>=0;i--)ctx.lineTo(X(D.x[i]),Y(B[i]));ctx.closePath();ctx.fillStyle=color;ctx.fill();}
function escena(t){ctx.clearRect(0,0,W,H);ejes();
  ctx.save();ctx.beginPath();ctx.rect(M.l,M.t-6,W-M.l-M.r+2,H-M.t-M.b+12);ctx.clip();
  const fin=t>3000?Math.min(1,(t-3000)/700):0;
  // 1) lluvia de caminos
  let visibles=0;
  D.caminos.forEach((c,i)=>{const p=ease((t-i*10)/1900);if(p<=0)return;visibles++;
    ctx.strokeStyle=D.sube[i]?CORAL:ACERO;ctx.globalAlpha=(p<1?.55:.32)*(1-fin*.72);ctx.lineWidth=p<1?1.4:1;
    const h=camino(c,p);ctx.stroke();
    if(p<1){ctx.globalAlpha=.9;ctx.fillStyle=D.sube[i]?CORAL:ACERO;ctx.beginPath();ctx.arc(h[0],h[1],2.2,0,7);ctx.fill();}});
  ctx.globalAlpha=1;
  const cnt=Math.min(N_TOTAL,Math.round(N_TOTAL*Math.min(1,visibles/D.caminos.length)*ease(t/2600)));
  document.getElementById('cont').textContent=cnt.toLocaleString('es-CO')+' escenarios simulados';
  // 2) bandas de probabilidad
  const b=ease((t-2700)/800);
  if(b>0){banda('5','95',`rgba(255,107,91,${.13*b})`);banda('25','75',`rgba(255,107,91,${.22*b})`);}
  // 3) mediana que se dibuja con brillo
  const m=easeIO((t-2900)/1000);
  if(m>0){ctx.save();ctx.shadowColor='rgba(244,237,224,.8)';ctx.shadowBlur=12;ctx.strokeStyle=CREMA;ctx.lineWidth=2.6;camino(D.bandas['50'],m);ctx.stroke();ctx.restore();
    ctx.save();ctx.globalAlpha=.85*m;ctx.strokeStyle='rgba(255,156,143,.9)';ctx.setLineDash([3,4]);ctx.lineWidth=1.2;
    camino(D.bandas['95'],m);ctx.stroke();camino(D.bandas['5'],m);ctx.stroke();ctx.restore();}
  ctx.restore();
  lineaH(D.s0,'#D9D2C4','Hoy: '+fmt(D.s0)+' '+MON,.9);
  if(D.objetivo)lineaH(D.objetivo,'#FFD6A0','Tu objetivo: '+fmt(D.objetivo),Math.min(1,t/800));
  // 4) histograma de precios finales, de lado
  const hg=ease((t-3300)/900);
  if(hg>0){const x0=W-M.r+10,ancho=M.r-92,mx=Math.max(...D.hist);
    for(let i=0;i<D.hist.length;i++){const y1=Y(D.bordes[i+1]),y0=Y(D.bordes[i]),l=ancho*D.hist[i]/mx*hg;
      const mid=(D.bordes[i]+D.bordes[i+1])/2;ctx.fillStyle=mid>=D.s0?'rgba(255,107,91,.75)':'rgba(125,155,209,.75)';
      ctx.fillRect(x0,y1+.6,l,Math.max(1,y0-y1-1.2));}
    ctx.fillStyle='#9AA3B8';ctx.font='11px Inter, sans-serif';ctx.textAlign='left';ctx.fillText('Precio final',x0,M.t-12);}
  // 5) etiquetas finales y punto que late
  const lb=ease((t-3800)/600);
  if(lb>0){const xe=X(1),et=[[D.p95,'P95 · optimista',CORAL_SUAVE()],[D.mediana,'Precio objetivo',CREMA],[D.p5,'P5 · pesimista',ACERO]];
    ctx.save();ctx.globalAlpha=lb;let ult=-99;
    et.sort((a,b)=>b[0]-a[0]).forEach(([v,txt,col])=>{let y=Y(v);if(y-ult<30)y=ult+30;ult=y;const xl=W-80;
      ctx.fillStyle=col;ctx.font='600 13px "Inter Tight", Inter, sans-serif';ctx.textAlign='left';ctx.fillText(fmt(v),xl,y+1);
      ctx.font='10.5px Inter, sans-serif';ctx.fillStyle='#9AA3B8';ctx.fillText(txt,xl,y+14);});
    ctx.restore();}
}
function pulso(t){const lb=ease((t-3800)/600);if(lb<=0)return;const xe=X(1),pul=(Math.sin(t/380)+1)/2,ym=Y(D.mediana);ctx.save();ctx.globalAlpha=lb;
  ctx.fillStyle='rgba(244,237,224,'+(.25*(1-pul))+')';ctx.beginPath();ctx.arc(xe,ym,6+10*pul,0,7);ctx.fill();
  ctx.fillStyle=CREMA;ctx.beginPath();ctx.arc(xe,ym,5,0,7);ctx.fill();ctx.restore();}
function cursor(t){
  if(hover!==null&&t>4200){const f=Math.max(0,Math.min(1,(hover-M.l)/(W-M.l-M.r)));const n=D.x.length;let j=0;while(j<n-1&&D.x[j+1]<f)j++;
    const xx=X(D.x[j]);ctx.save();ctx.strokeStyle='rgba(244,237,224,.45)';ctx.beginPath();ctx.moveTo(xx,M.t);ctx.lineTo(xx,H-M.b);ctx.stroke();
    const filas=[['P95',D.bandas['95'][j],CORAL],['P75',D.bandas['75'][j],'#FF9C8F'],['Mediana',D.bandas['50'][j],CREMA],['P25',D.bandas['25'][j],'#AFC2E6'],['P5',D.bandas['5'][j],ACERO]];
    const bw=170,bh=24+filas.length*18,bx=xx+12+bw>W-M.r?xx-bw-12:xx+12,by=M.t+6;
    ctx.fillStyle='rgba(14,24,48,.92)';ctx.strokeStyle='rgba(244,237,224,.22)';ctx.beginPath();ctx.roundRect(bx,by,bw,bh,10);ctx.fill();ctx.stroke();
    ctx.font='600 12px Inter, sans-serif';ctx.fillStyle=CREMA;ctx.textAlign='left';ctx.fillText(etiquetaX(D.x[j]),bx+12,by+18);
    filas.forEach(([k,v,c],i)=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(bx+16,by+34+i*18,3.5,0,7);ctx.fill();ctx.fillStyle='#D9D2C4';ctx.font='12px Inter, sans-serif';
      ctx.fillText(k,bx+26,by+38+i*18);ctx.textAlign='right';ctx.fillText(fmt(v),bx+bw-12,by+38+i*18);ctx.textAlign='left';});
    ctx.restore();}
}
function CORAL_SUAVE(){return '#FF9C8F'}
let cache=null;const FIN=4500;
function bucle(){const t=reduce?1e9:performance.now()-t0;
  if(t<FIN){escena(t);}
  else{if(!cache){escena(FIN);cache=document.createElement('canvas');cache.width=cv.width;cache.height=cv.height;cache.getContext('2d').drawImage(cv,0,0);}
    ctx.save();ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(cache,0,0);ctx.restore();}
  pulso(reduce?FIN:t);cursor(t);requestAnimationFrame(bucle);}
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();hover=e.clientX-r.left;});
cv.addEventListener('mouseleave',()=>{hover=null;});
document.getElementById('rep').onclick=()=>{t0=performance.now();cache=null;};
window.addEventListener('resize',()=>{tam();cache=null;});tam();requestAnimationFrame(bucle);
</script></body></html>"""


def html_simulacion(datos: dict, titulo: str, moneda: str, n_total: int, alto: int = 470) -> str:
    return (PLANTILLA.replace("__DATOS__", json.dumps(datos, separators=(",", ":")))
            .replace("__TITULO__", titulo).replace("__MONEDA__", moneda)
            .replace("__NTOTAL__", str(int(n_total))).replace("__ALTO__", str(alto)))
