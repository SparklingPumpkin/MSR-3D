'use strict';
const results = {
  lerf45: { description: 'Three target scenes · 45 exact-string-unseen queries', rows: [[59.69,76.20,65.68],[56.29,71.75,66.19],[63.15,83.21,72.83]] },
  lerf67: { description: 'Four scenes · 67 queries, including the supervised Waldo Kitchen scene', rows: [[58.01,75.20,61.07],[50.74,64.88,58.93],[64.70,85.52,72.63]] },
  '3dovs61': { description: 'Ten target scenes · 61 scene/query pairs · geometric middle-crop anchors', rows: [[79.41,93.33,80.52],[76.12,90.48,81.11],[80.98,95.00,83.02]] }
};
for (const button of document.querySelectorAll('[data-cohort]')) {
  button.addEventListener('click', () => {
    for (const other of document.querySelectorAll('[data-cohort]')) other.setAttribute('aria-pressed', String(other === button));
    const result = results[button.dataset.cohort];
    document.getElementById('cohort-description').textContent = result.description;
    document.querySelectorAll('#results-body tr').forEach((row, index) => {
      row.querySelectorAll('td').forEach((cell, column) => { cell.textContent = result.rows[index][column].toFixed(2); });
    });
  });
}
const slider = document.getElementById('comparison-slider');
slider.addEventListener('input', () => {
  document.getElementById('comparison').style.setProperty('--split', `${slider.value}%`);
  slider.setAttribute('aria-valuetext', `${slider.value} percent selected view`);
});
async function loadGallery() {
  const response = await fetch('assets/gallery.json');
  if (!response.ok) throw new Error('Gallery unavailable');
  const items = await response.json();
  let selection = 0;
  const original = document.getElementById('original-image');
  const selected = document.getElementById('selected-image');
  const tabs = document.getElementById('query-tabs');
  const setItem = async (item, button) => {
    const version = ++selection;
    document.getElementById('gallery-status').textContent = 'Loading example…';
    const before = new Image(); const after = new Image();
    before.src = item.original; after.src = item.selected;
    try { await Promise.all([before.decode(), after.decode()]); }
    catch { if (version === selection) document.getElementById('gallery-status').textContent = 'This example could not load. Please try again.'; return; }
    if (version !== selection) return;
    original.src = before.src; original.alt = `Original benchmark view of ${item.scene.replaceAll('_', ' ')}`;
    selected.src = after.src; selected.alt = item.alt;
    document.getElementById('comparison').style.aspectRatio = `${before.naturalWidth} / ${before.naturalHeight}`;
    document.getElementById('scene-name').textContent = item.scene.replaceAll('_', ' ');
    document.getElementById('query-name').textContent = `“${item.query}”`;
    for (const tab of tabs.children) tab.setAttribute('aria-pressed', String(tab === button));
    document.getElementById('gallery-status').textContent = '';
  };
  items.forEach((item, index) => {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = `“${item.query}”`; button.setAttribute('aria-pressed','false');
    button.addEventListener('click', () => { void setItem(item, button); });
    tabs.append(button);
    if (index === 0) void setItem(item, button);
  });
}
loadGallery().catch(() => { document.getElementById('gallery-status').textContent = 'Examples could not load. Please refresh the page.'; });
document.getElementById('copy-citation').addEventListener('click', async () => {
  const text = document.getElementById('bibtex').textContent;
  const status = document.getElementById('copy-status');
  try { await navigator.clipboard.writeText(text); status.textContent = 'Citation copied.'; }
  catch {
    const selection = window.getSelection();
    const range = document.createRange(); range.selectNodeContents(document.getElementById('bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Use your device’s copy command.';
  }
});
