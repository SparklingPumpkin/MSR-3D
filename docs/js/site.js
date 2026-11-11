// MSR-3D page: play clips only while visible, bucket chart toggle, BibTeX copy, active nav link.
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const io = new IntersectionObserver(entries => {
    for (const en of entries) {
      const v = en.target;
      if (en.isIntersecting && !reduce) { v.preload = 'auto'; v.play().catch(() => {}); } else v.pause();
    }
  }, {rootMargin: '120px 0px', threshold: 0.15});
  document.querySelectorAll('video[data-autoplay]').forEach(v => io.observe(v));

  document.querySelectorAll('[data-bucket]').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('[data-bucket]').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    document.querySelectorAll('[data-bucket-panel]').forEach(p => { p.hidden = p.dataset.bucketPanel !== b.dataset.bucket; });
  }));

  document.querySelectorAll('[data-copy]').forEach(b => b.addEventListener('click', async () => {
    const text = document.querySelector(b.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; }
    catch { getSelection().selectAllChildren(document.querySelector(b.dataset.copy)); b.textContent = 'Press Ctrl+C'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1800);
  }));

  const links = [...document.querySelectorAll('.nav-links a')];
  const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const nav = new IntersectionObserver(entries => {
    for (const en of entries) if (en.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + en.target.id));
  }, {rootMargin: '-45% 0px -50% 0px'});
  targets.forEach(t => nav.observe(t));
})();
