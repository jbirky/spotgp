// Append a "Download notebook" link to the bottom of the right-side
// "On this page" TOC column for notebook-based tutorial pages.
document.addEventListener("DOMContentLoaded", function () {
  var meta = document.querySelector('meta[name="notebook-source"]');
  if (!meta) return;

  var filename = meta.getAttribute("content");
  if (!filename) return;

  // Furo's right-side TOC lives inside <aside class="toc-drawer">
  var tocDrawer = document.querySelector(".toc-drawer");
  if (!tocDrawer) return;

  var wrapper = document.createElement("div");
  wrapper.className = "nb-download-link";

  var link = document.createElement("a");
  link.href = filename;
  link.download = "";
  link.textContent = "\u2B07 Download notebook";

  wrapper.appendChild(link);
  tocDrawer.insertBefore(wrapper, tocDrawer.firstChild);
});
