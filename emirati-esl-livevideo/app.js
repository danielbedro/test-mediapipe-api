// Only MVP
// Pipeline: Webcam -> MediaPipe Hands -> Landmark features -> KNN -> Smoothing -> Decoder -> Text

// ---- DOM
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const hud = {
  handsCount: document.getElementById('handsCount'),
  fps: document.getElementById('fps'),
  status: document.getElementById('status'),
  currentPred: document.getElementById('currentPred'),
  stablePred: document.getElementById('stablePred'),
};

const btnStartCam = document.getElementById('startCam');
const btnStopCam = document.getElementById('stopCam');
const btnStartRecog = document.getElementById('startRecog');
const btnStopRecog = document.getElementById('stopRecog');
const inputK = document.getElementById('knnK');
const inputSmoothN = document.getElementById('smoothN');
const inputCommitN = document.getElementById('commitN');
const inputCooldownMs = document.getElementById('cooldownMs');

const labelInput = document.getElementById('labelInput');
const btnCollectOne = document.getElementById('collectOne');
const btnBurst2s = document.getElementById('burst2s');
const btnClearLabel = document.getElementById('clearLabel');
const btnExport = document.getElementById('exportData');
const inputImport = document.getElementById('importData');
const btnClearAll = document.getElementById('clearAll');
const textTotalSamples = document.getElementById('totalSamples');
const labelCountsDiv = document.getElementById('labelCounts');

const arabicMode = document.getElementById('arabicMode');
const speakOut = document.getElementById('speakOut');
const btnBackspace = document.getElementById('backspace');
const btnSpace = document.getElementById('space');
const btnClearText = document.getElementById('clearText');
const output = document.getElementById('output');

// ---- State
let camera = null;
let running = false;
let recognizing = false;
let lastFrameTime = performance.now();
let smoothQueue = [];
let stableQueue = [];
let cooldownUntil = 0;
let latestLandmarks = null;
let lastBBox = null;

// ---- KNN implementation (put this BEFORE creating `knn`) ----
function cosineSim(a, b) { let dot = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8); }
class SimpleKNN {
    constructor(k = 5) { this.k = k; this.samples = []; }
    setK(k) { this.k = Math.max(1, k | 0); }
    clear() { this.samples = []; }
    addSample(x, label) { this.samples.push({ x: new Float32Array(x), label }); }
    bulkLoad(samples) { this.samples = samples.map(s => ({ x: new Float32Array(s.x), label: s.label })); }
    count() { return this.samples.length; }
    predict(x) {
        if (this.samples.length === 0) return null;
        const v = (x instanceof Float32Array) ? x : new Float32Array(x);
        const sims = this.samples.map(s => ({ sim: cosineSim(v, s.x), label: s.label }));
        sims.sort((a, b) => b.sim - a.sim);
        const k = Math.min(this.k, sims.length);
        const top = sims.slice(0, k);
        const tally = {}; let simSum = 0;
        for (const t of top) { tally[t.label] = (tally[t.label] || 0) + 1; simSum += t.sim; }
        let best = null, bestCount = -1;
        for (const [lab, c] of Object.entries(tally)) { if (c > bestCount) { best = lab; bestCount = c; } }
        const avgSim = simSum / k;
        return { label: best, avgSim, votes: bestCount, k };
    }
}

// Dataset: array of { x: Float32Array(features), label: string }
let dataset = loadDatasetLocal() || { samples: [] };
const knn = new SimpleKNN(parseInt(inputK.value, 10));
refreshDatasetStats();



// ---- Features from landmarks
function landmarksToFeature(landmarks){
  // normalize by 2D bbox (x,y), keep z as-is; flip y to make up positive
  let minX=1, minY=1, maxX=0, maxY=0;
  for(const p of landmarks){ if(p.x<minX)minX=p.x; if(p.y<minY)minY=p.y; if(p.x>maxX)maxX=p.x; if(p.y>maxY)maxY=p.y; }
  const w = Math.max(1e-4,maxX-minX), h=Math.max(1e-4,maxY-minY);
  const feat = new Float32Array(21*3);
  for(let i=0;i<21;i++){
    const p=landmarks[i];
    const nx = (p.x-minX)/w - 0.5;
    const ny = -((p.y-minY)/h - 0.5);
    const nz = p.z || 0;
    feat[i*3+0]=nx; feat[i*3+1]=ny; feat[i*3+2]=nz;
  }
  return feat;
}
function bbox2D(landmarks){
  let minX=1,minY=1,maxX=0,maxY=0;
  for(const p of landmarks){ if(p.x<minX)minX=p.x; if(p.y<minY)minY=p.y; if(p.x>maxX)maxX=p.x; if(p.y>maxY)maxY=p.y; }
  return {minX,minY,maxX,maxY,w:(maxX-minX),h:(maxY-minY)};
}
function bboxMotion(prev, now){
  if(!prev||!now) return 0;
  const dx = (now.minX - prev.minX) + (now.maxX - prev.maxX);
  const dy = (now.minY - prev.minY) + (now.maxY - prev.maxY);
  return Math.hypot(dx,dy);
}

// ---- MediaPipe setup
const hands = new Hands({
    locateFile: (file) => `https://unpkg.com/@mediapipe/hands/${file}`});
hands.setOptions({ maxNumHands:2, modelComplexity:1, minDetectionConfidence:0.6, minTrackingConfidence:0.5 });
hands.onResults(onResults);

function syncCanvas(){ const w=video.videoWidth||960, h=video.videoHeight||720; if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h;} }
function drawLandmarks(results){
  syncCanvas();
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!results.multiHandLandmarks) return;
  for(const lm of results.multiHandLandmarks){
    window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, {color:'#00ff6a', lineWidth:2});
    window.drawLandmarks(ctx, lm, {color:'#00ff6a', radius:2.5});
  }
}

function onResults(results){
  const now = performance.now();
  const dt = (now-lastFrameTime)/1000; lastFrameTime = now;
  hud.fps.textContent = Math.max(1,(1/dt)|0);
  hud.handsCount.textContent = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;

  drawLandmarks(results);

  // choose one hand (right-most in mirrored view)
  if(results.multiHandLandmarks && results.multiHandLandmarks.length){
    const handsLms = results.multiHandLandmarks.slice().sort((a,b)=>{
      const ax=a.reduce((s,p)=>s+p.x,0)/a.length;
      const bx=b.reduce((s,p)=>s+p.x,0)/b.length;
      return bx-ax; // right-most first (mirrored video)
    });
    latestLandmarks = handsLms[0];
  } else {
    latestLandmarks = null;
    hud.currentPred.textContent = '—';
  }

  if(recognizing) stepRecognition();
}

// ---- Camera
async function startCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({ video:{width:960,height:720} });
  video.srcObject = stream; await video.play();
  camera = new Camera(video, { onFrame: async()=>{ await hands.send({ image: video }); }, width:960, height:720 });
  camera.start(); running=true; hud.status.textContent='Camera running';
}
function stopCamera(){
  if(camera){ camera.stop(); camera=null; }
  if(video.srcObject){ video.srcObject.getTracks().forEach(t=>t.stop()); video.srcObject=null; }
  running=false; hud.status.textContent='Camera stopped';
}

// ---- Recognition loop step
function stepRecognition(){
  if(!latestLandmarks || knn.count()===0){ hud.currentPred.textContent='—'; return; }
  const feat = landmarksToFeature(latestLandmarks);
  const pred = knn.predict(feat);
  const label = pred?.label || null;

  hud.currentPred.textContent = label ? `${label}` : '—';

  // temporal smoothing & stability
  const N = parseInt(inputSmoothN.value,10);
  smoothQueue.push(label); if(smoothQueue.length>N) smoothQueue.shift();
  const smoothed = majority(smoothQueue);

  const bb = bbox2D(latestLandmarks);
  const mot = bboxMotion(lastBBox, bb);
  lastBBox = bb;

  const now = performance.now();
  const commitN = parseInt(inputCommitN.value,10);
  const cooldownMs = parseInt(inputCooldownMs.value,10);

  if(smoothed){
    stableQueue.push(smoothed);
    if(stableQueue.length>commitN) stableQueue.shift();
  } else {
    stableQueue = [];
  }

  const stable = (stableQueue.length===commitN && allEqual(stableQueue));
  const still = mot < 0.01; // motion threshold in normalized coords
  const coolOK = now >= cooldownUntil;

  hud.stablePred.textContent = stable ? smoothed : '—';

  if(recognizing && stable && still && coolOK){
    commitLetter(smoothed);
    stableQueue = [];
    cooldownUntil = now + cooldownMs;
  }
}

// ---- Decoder & Output
function commitLetter(lbl){
  if(!lbl) return;
  const text = output.value || '';
  const mapped = arabicMode.checked ? mapToArabic(lbl) : lbl;
  output.value = text + mapped;
  if(speakOut.checked){
    try{ const u = new SpeechSynthesisUtterance(mapped); u.lang = arabicMode.checked ? 'ar-AE' : 'en-US'; speechSynthesis.speak(u);}catch(e){}
  }
}
function mapToArabic(lbl){
  // If you label classes with Latin (A,B,...) but want Arabic, map here.
  // Placeholder passthrough — customize this table to your ESL alphabet set.
  const map = {
    'A':'ا','B':'ب','T':'ت','TH':'ث','J':'ج','H':'ح','KH':'خ','D':'د','TH_2':'ذ',
    'R':'ر','Z':'ز','S':'س','SH':'ش','SAD':'ص','DAD':'ض','TTA':'ط','ZHA':'ظ',
    'AIN':'ع','GHAIN':'غ','F':'ف','Q':'ق','K':'ك','L':'ل','M':'م','N':'ن',
    'H_2':'ه','W':'و','Y':'ي'
  };
  return map[lbl] || lbl;
}
function majority(arr){
  if(!arr.length) return null;
  const tally = {};
  for(const x of arr){ if(!x) continue; tally[x]=(tally[x]||0)+1; }
  let best=null, cnt=-1; for(const [k,v] of Object.entries(tally)){ if(v>cnt){best=k;cnt=v;} }
  return best;
}
function allEqual(arr){ if(!arr.length) return false; return arr.every(x=>x===arr[0]); }

// ---- Dataset ops
function saveDatasetLocal(){
  try{ localStorage.setItem('eslDataset', JSON.stringify(dataset)); }catch(e){}
}
function loadDatasetLocal(){
  try{ const s = localStorage.getItem('eslDataset'); return s ? JSON.parse(s) : null; }catch(e){return null;}
}
function refreshDatasetStats(){
  textTotalSamples.textContent = dataset.samples.length;
  const counts = {};
  for(const s of dataset.samples){ counts[s.label]=(counts[s.label]||0)+1; }
  labelCountsDiv.innerHTML = '';
  Object.keys(counts).sort().forEach(k=>{
    const span = document.createElement('span');
    span.className='badge';
    span.textContent = `${k}: ${counts[k]}`;
    labelCountsDiv.appendChild(span);
  });
  // Refresh KNN memory
  knn.clear();
  for(const s of dataset.samples){ knn.addSample(s.x, s.label); }
}

function addSample(label, lms){
  if(!label) return;
  const feat = landmarksToFeature(lms);
  dataset.samples.push({ x: Array.from(feat), label });
  saveDatasetLocal();
  refreshDatasetStats();
}

// ---- UI handlers
btnStartCam.onclick = ()=>startCamera().catch(err=>alert('Camera error: '+err.message));
btnStopCam.onclick = ()=>stopCamera();

btnStartRecog.onclick = ()=>{ recognizing=true; hud.status.textContent='Recognizing...'; };
btnStopRecog .onclick = ()=>{ recognizing=false; hud.status.textContent='Paused'; };

inputK.oninput = ()=>{ knn.setK(parseInt(inputK.value,10)); };
inputSmoothN.oninput = ()=>{ smoothQueue = []; };
inputCommitN.oninput = ()=>{ stableQueue = []; };

btnCollectOne.onclick = ()=>{
  const lab = labelInput.value.trim();
  if(!lab) return alert('Enter a label first.');
  if(!latestLandmarks) return alert('No hand detected.');
  addSample(lab, latestLandmarks);
};

btnBurst2s.onclick = async ()=>{
  const lab = labelInput.value.trim();
  if(!lab) return alert('Enter a label first.');
  if(!running) return alert('Start the camera first.');
  hud.status.textContent='Recording burst...';
  const end = performance.now()+2000;
  let added=0;
  while(performance.now()<end){
    if(latestLandmarks){ addSample(lab, latestLandmarks); added++; }
    await sleep(60); // ~16 fps burst on average landmark frames
  }
  hud.status.textContent=`Added ${added} samples to ${lab}`;
};

btnClearLabel.onclick = ()=>{
  const lab = labelInput.value.trim();
  if(!lab) return alert('Enter a label to clear.');
  dataset.samples = dataset.samples.filter(s=>s.label!==lab);
  saveDatasetLocal(); refreshDatasetStats();
};

btnExport.onclick = ()=>{
  const blob = new Blob([JSON.stringify(dataset)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download='esl_dataset.json'; a.click(); URL.revokeObjectURL(url);
};

inputImport.onchange = (e)=>{
  const file = e.target.files?.[0]; if(!file) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    try{
      const data = JSON.parse(reader.result);
      if(!data.samples) throw new Error('Invalid dataset format');
      dataset = data; saveDatasetLocal(); refreshDatasetStats();
      alert('Dataset imported.');
    }catch(err){ alert('Failed: '+err.message); }
  };
  reader.readAsText(file);
};

btnClearAll.onclick = ()=>{
  if(!confirm('Clear ALL samples?')) return;
  dataset = { samples: [] };
  saveDatasetLocal(); refreshDatasetStats();
};

btnBackspace.onclick = ()=>{ output.value = output.value.slice(0,-1); };
btnSpace.onclick = ()=>{ output.value += ' '; };
btnClearText.onclick = ()=>{ output.value=''; };

// ---- helpers
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

// ---- Start-up hint
hud.status.textContent = 'Idle — click "Start Camera"';

// ===== END app.js =====
