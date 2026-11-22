// Optional object-aware mode; existing baseline question form remains unchanged.
(() => {
  const form = document.getElementById('questionForm');
  const question = document.getElementById('questionInput');
  const select = document.getElementById('videoSelect');
  if (!form || !question || !select) return;
  const panel = document.createElement('div');
  panel.style.margin = '12px 0';
  const toggle = document.createElement('input');
  toggle.type = 'checkbox'; toggle.id = 'objectMode';
  const label = document.createElement('label'); label.htmlFor='objectMode';
  label.textContent=' Object-aware mode (experimental; slower)';
  const phrase = document.createElement('input');
  phrase.type='text';phrase.placeholder='Object to track, e.g. toy car';
  phrase.maxLength=120;phrase.style.display='none';phrase.style.width='100%';
  phrase.style.marginTop='8px';
  toggle.addEventListener('change',()=>{phrase.style.display=toggle.checked?'block':'none';});
  panel.append(toggle,label,phrase);form.prepend(panel);
  // Capture only object-mode submissions; let existing app.js handle baseline.
  form.addEventListener('submit',async event=>{
    if(!toggle.checked)return;
    event.preventDefault();event.stopImmediatePropagation();
    const object_phrase=phrase.value.trim();
    if(!object_phrase){alert('Enter the object to track.');return;}
    const video_id=select.value;
    if(!video_id){alert('Select an indexed video.');return;}
    const btn=form.querySelector('button[type="submit"]') || form.querySelector('button');
    if(btn)btn.disabled=true;
    const messages=document.getElementById('messages');
    const status=document.createElement('p');status.textContent='Running object-aware query…';
    messages?.appendChild(status);
    try{
      const res=await fetch(`/api/videos/${encodeURIComponent(video_id)}/query-object`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question:question.value,object_phrase})});
      const data=await res.json();
      if(!res.ok)throw new Error(data.detail || `HTTP ${res.status}`);
      status.textContent=`${data.answer}\nObject status: ${data.object_status}; object frames: ${data.object_evidence_count}`;
      status.style.whiteSpace='pre-wrap';
      const evidence=document.getElementById('evidence');
      if(evidence){evidence.textContent='Evidence timestamps: '+data.frames.map(f=>`${Number(f.timestamp).toFixed(2)}s (${f.evidence_role})`).join(', ');}
    }catch(err){status.textContent=`Object query error: ${err.message}`;}
    finally{if(btn)btn.disabled=false;}
  },true);
})();
