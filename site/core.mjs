
const n=(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback;
const clamp=(value,low,high)=>Math.min(high,Math.max(low,value));
const round=(value,digits=2)=>Number(value.toFixed(digits));
const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;
const parseJSON=(value,fallback=[])=>{try{return JSON.parse(value)}catch{return fallback}};
const valuesFrom=value=>String(value).split(/[\s,]+/).map(Number).filter(Number.isFinite);
const result=(status,summary,metrics,rows,detail='')=>({status,summary,metrics,rows,detail});
const erf=x=>{const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+0.3275911*a);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a);return sign*y};
const normalCdf=z=>0.5*(1+erf(z/Math.sqrt(2)));
const wilson=(successes,total)=>{if(!total)return[0,0];const z=1.96,p=successes/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,h=z*Math.sqrt((p*(1-p)+z*z/(4*total))/total)/d;return[clamp(c-h,0,1),clamp(c+h,0,1)]};
const sha256=async value=>{const bytes=new TextEncoder().encode(String(value));const digest=await crypto.subtle.digest('SHA-256',bytes);return[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')};
const tag=(xml,name)=>xml.match(new RegExp('<'+name+'[^>]*>([\\s\\S]*?)<\\/'+name+'>','i'))?.[1]?.trim()??'';
const similarity=(a,b)=>{const x=String(a).toLowerCase(),y=String(b).toLowerCase();if(x===y)return 1;const A=new Set(x.split(/\W+/).filter(Boolean)),B=new Set(y.split(/\W+/).filter(Boolean));const inter=[...A].filter(v=>B.has(v)).length;return inter/Math.max(1,new Set([...A,...B]).size)};

export const meta={"slug":"upliftpolicy","name":"Uplift Policy","eyebrow":"Retention targeting","description":"Target customers by incremental treatment effect while excluding sleeping dogs.","fields":[{"name":"capacity","label":"Contact capacity","type":"number","min":1,"max":100,"step":1,"help":""},{"name":"valuePerSave","label":"Value per retained customer","type":"number","min":1,"max":10000,"step":10,"help":""},{"name":"customers","label":"Customer uplift estimates as JSON","type":"textarea","rows":10,"help":""}]};
export const initialState={"capacity":3,"valuePerSave":500,"customers":"[{\"id\":\"C01\",\"uplift\":0.18,\"risk\":0.72},{\"id\":\"C02\",\"uplift\":-0.09,\"risk\":0.84},{\"id\":\"C03\",\"uplift\":0.12,\"risk\":0.61},{\"id\":\"C04\",\"uplift\":0.04,\"risk\":0.91},{\"id\":\"C05\",\"uplift\":0.15,\"risk\":0.55}]"};
export const alternateState={"capacity":1,"valuePerSave":500,"customers":"[{\"id\":\"C01\",\"uplift\":0.18,\"risk\":0.72},{\"id\":\"C02\",\"uplift\":-0.09,\"risk\":0.84},{\"id\":\"C03\",\"uplift\":0.12,\"risk\":0.61},{\"id\":\"C04\",\"uplift\":0.04,\"risk\":0.91},{\"id\":\"C05\",\"uplift\":0.15,\"risk\":0.55}]"};
export async function compute(i){const cap=Math.max(0,Math.round(n(i.capacity))),value=n(i.valuePerSave),customers=parseJSON(i.customers,[]),ranked=[...customers].filter(c=>n(c.uplift)>0).sort((a,b)=>n(b.uplift)-n(a.uplift)),selected=new Set(ranked.slice(0,cap).map(c=>c.id)),rows=customers.map(c=>({customer:c.id,uplift:`${round(n(c.uplift)*100)}%`,risk:`${round(n(c.risk)*100)}%`,decision:n(c.uplift)<0?'Protect from contact':selected.has(c.id)?'Contact':'Hold out'})),expected=customers.filter(c=>selected.has(c.id)).reduce((s,c)=>s+n(c.uplift)*value,0),dogs=customers.filter(c=>n(c.uplift)<0).length;return result('Policy evaluated',`${selected.size} customers selected; ${dogs} harmful-contact cases excluded.`,[{label:'Expected policy value',value:`$${Math.round(expected).toLocaleString()}`},{label:'Customers contacted',value:selected.size},{label:'Sleeping dogs protected',value:dogs}],rows,'Ranking uses incremental uplift, not churn risk alone.')}
