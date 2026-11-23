// Optional object-aware mode; existing baseline question form remains unchanged.
(() => {
  const form = document.getElementById('questionForm');
  const question = document.getElementById('questionInput');
  const select = document.getElementById('videoSelect');
  const toggle = document.getElementById('objectMode');
  const controls = document.getElementById('objectReviewControls');
  const phrase = document.getElementById('objectPhrase');
  const reviewMode = document.getElementById('objectReviewMode');
  if (!form || !question || !select || !toggle || !controls || !phrase || !reviewMode) return;
  function updateControls() {
    controls.hidden = !toggle.checked;
  }
  toggle.addEventListener('change', updateControls);
  updateControls();
  // Capture only object-mode submissions; let existing app.js handle baseline.
  form.addEventListener('submit',async event=>{
    if(!toggle.checked)return;
    event.preventDefault();event.stopImmediatePropagation();
    const object_phrase=phrase.value.trim();
    if(!object_phrase){alert('Enter the object to track.');return;}
    const mode = reviewMode.value;
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
        body:JSON.stringify({question:question.value, object_phrase, review_mode:mode})});
      const data=await res.json();
      if(!res.ok)throw new Error(data.detail || `HTTP ${res.status}`);
      if (mode === 'events' && !Array.isArray(data.events)) {
        throw new Error('The server returned an Object summary. Restart TEMPO from the updated folder.');
      }
      status.textContent=`${data.answer}\nObject status: ${data.object_status}; object frames: ${data.object_evidence_count}`;
      status.style.whiteSpace='pre-wrap';
      const evidence=document.getElementById('evidence');
      if (evidence && select.value === video_id) {
        evidence.replaceChildren();
        const cited = new Set(data.cited_frame_ids);
        const byId = new Map(data.frames.map(frame => [frame.frame_id, frame]));
        function appendTimestamps(row, ids) {
          for (const id of ids) {
            const frame = byId.get(id);
            if (!frame) continue;
            const link = document.createElement('button');
            link.type = 'button';
            link.textContent = `${frame.timestamp.toFixed(2)} s`;
            link.addEventListener('click', () => {
              if (select.value !== video_id) return;
              const player = document.getElementById('videoPlayer');
              if (player) {
                player.pause();
                player.currentTime = frame.timestamp;
              }
            });
            row.append(link, ' ');
          }
        }
        if (Array.isArray(data.events)) {
          for (const event of data.events) {
            const row = document.createElement('p');
            row.append(`${event.description} `);
            appendTimestamps(row, event.frame_ids);
            evidence.appendChild(row);
          }
          if (data.sighting_frame_ids?.length) {
            const row = document.createElement('p');
            row.append('Object sightings ');
            appendTimestamps(row, data.sighting_frame_ids);
            evidence.appendChild(row);
          } else if (!data.events.length) {
            evidence.textContent = 'No cited events or sightings.';
          }
        } else {
          const frames = data.frames.filter(frame => cited.has(frame.frame_id));
          evidence.textContent = frames.length
            ? 'Cited timestamps: ' + frames.map(frame => `${frame.timestamp.toFixed(2)} s`).join(', ')
            : 'No cited frames.';
        }
      }
    }catch(err){status.textContent=`Object query error: ${err.message}`;}
    finally{if(btn)btn.disabled=false;}
  },true);
})();
