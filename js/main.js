/* ============================================================
   LIBERO-Recover project page — interactions (v2)
   Hero video reel, tabbed level widget, flip cards, figure
   deck carousel, scroll progress, hover-to-play videos,
   suite filter, nav highlighting, reveal, BibTeX copy.
   Honors prefers-reduced-motion.
   ============================================================ */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- scroll progress ---------- */

  var bar = document.getElementById("scrollBar");
  function updateProgress() {
    if (!bar) return;
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
  }

  /* ---------- hero reel: duplicate each row, then play ---------- */

  var reel = document.getElementById("heroReel");
  if (reel && !reduceMotion) {
    reel.querySelectorAll(".reel-row").forEach(function (row) {
      row.innerHTML += row.innerHTML; /* seamless -50% marquee loop */
    });
    var reelVideos = reel.querySelectorAll("video");
    if ("IntersectionObserver" in window) {
      var rio = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          reelVideos.forEach(function (v) {
            if (en.isIntersecting) {
              v.play().catch(function () { /* autoplay blocked — dark tiles */ });
            } else {
              v.pause();
            }
          });
        });
      }, { rootMargin: "60px" });
      rio.observe(reel);
    }
  }

  /* ---------- generic hover-to-play video boxes ---------- */

  document.querySelectorAll("[data-video]").forEach(function (box) {
    var video = box.querySelector("video");
    if (!video) return;

    var pinned = false;
    function start() {
      video.play().then(function () {
        box.classList.add("playing");
      }).catch(function () { /* autoplay blocked — poster stays */ });
    }
    function stop() {
      if (pinned) return;
      video.pause();
      box.classList.remove("playing");
    }
    function toggle() {
      pinned = !pinned;
      if (pinned) { start(); }
      else { video.pause(); box.classList.remove("playing"); }
    }

    if (!reduceMotion) {
      box.addEventListener("mouseenter", start);
      box.addEventListener("mouseleave", stop);
    }
    box.addEventListener("click", toggle);
    box.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
    box.setAttribute("tabindex", "0");
    box.setAttribute("role", "button");
    box.setAttribute("aria-label", "Play demonstration video");
  });

  /* pause offscreen videos to keep the page light */
  if ("IntersectionObserver" in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target.querySelector("video");
        if (!v) return;
        if (!en.isIntersecting && !v.paused) {
          v.pause();
          en.target.classList.remove("playing");
        }
      });
    }, { rootMargin: "120px" });
    document.querySelectorAll("[data-video]").forEach(function (b) { vio.observe(b); });
  }

  /* ---------- level widget ---------- */

  var LEVELS = {
    l1: {
      badge: "L1 · Action Retry", name: "The plan still holds",
      blurb: 'The failure does not materially alter the task configuration — <span class="mono">s<sub>f</sub> ≈ s<sub>expected</sub></span>. Recovery only requires retrying the failed action, such as re-attempting a grasp on an unchanged object.',
      caps: ["closed-loop re-execution"], stat: "1 / 4"
    },
    l2: {
      badge: "L2 · Action Adaptation", name: "Adapt, don't repeat",
      blurb: 'The failure causes a limited change in object configuration, but the object remains directly usable. The model must adapt its action to the observed state rather than blindly repeating the failed action — <span class="mono">a′ = π(o<sub>f</sub>, l) ≠ a<sub>failed</sub></span>.',
      caps: ["observation-conditioned replanning"], stat: "2 / 4"
    },
    l3: {
      badge: "L3 · Object State Recovery", name: "Restore the object first",
      blurb: "The failure substantially alters a task-relevant object, making direct task completion impossible. The robot must first restore the object's required state — e.g., recover a displaced bowl's pose — and only then resume the instructed manipulation.",
      caps: ["object structure reasoning", "state reasoning"], stat: "3 / 4"
    },
    l4: {
      badge: "L4 · Environmental Recovery", name: "Clear the obstruction",
      blurb: "The failure affects an object or state not directly manipulated by the task but which subsequently prevents execution. Recovery requires reasoning about task-relevant <i>and</i> task-irrelevant states, their interaction topology, and restoring the environment before continuing.",
      caps: ["topological interaction reasoning", "scene memory"], stat: "4 / 4"
    }
  };

  var lvlTabs = document.querySelectorAll(".lvl-tab");
  var lvlStage = document.querySelector(".lvl-stage");
  var lvlBadge = document.getElementById("lvlBadge");
  var lvlName = document.getElementById("lvlName");
  var lvlBlurb = document.getElementById("lvlBlurb");
  var lvlCap = document.getElementById("lvlCap");
  var lvlStatV = document.getElementById("lvlStatV");

  function setLevel(key) {
    var d = LEVELS[key];
    if (!d || !lvlStage) return;

    lvlTabs.forEach(function (t) {
      var active = t.getAttribute("data-level") === key;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });

    /* stage accent + all derived colors follow --lvl-c */
    lvlStage.style.setProperty("--lvl-c", "var(--" + key + ")");

    lvlBadge.textContent = d.badge;
    lvlName.textContent = d.name;
    lvlBlurb.innerHTML = d.blurb;
    lvlCap.innerHTML = d.caps.map(function (c) {
      return '<span class="cap-chip">' + c + "</span>";
    }).join("");
    lvlStatV.textContent = d.stat;

    document.querySelectorAll(".lvl-video").forEach(function (v) {
      var mine = v.getAttribute("data-for") === key;
      v.hidden = !mine;
      if (!mine) { v.pause(); }
      else {
        v.setAttribute("preload", "metadata");
        if (!reduceMotion && lvlStage.querySelector(".lvl-stage-media:hover")) {
          v.play().catch(function () {});
        }
      }
    });
  }

  lvlTabs.forEach(function (t) {
    t.addEventListener("click", function () { setLevel(t.getAttribute("data-level")); });
  });
  if (lvlTabs.length) {
    setLevel(document.querySelector(".lvl-tab.is-active").getAttribute("data-level"));
  }

  /* ---------- suite flip cards: tap to flip ---------- */

  document.querySelectorAll(".suite-card").forEach(function (card) {
    card.addEventListener("click", function (e) {
      /* let the front video keep its own click-to-pin behavior */
      if (e.target.closest(".suite-media")) return;
      card.classList.toggle("is-flipped");
    });
    card.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        card.classList.toggle("is-flipped");
      }
    });
  });

  /* ---------- figure deck ---------- */

  var deck = document.getElementById("figdeck");
  if (deck) {
    var panels = deck.querySelectorAll(".figdeck-panel");
    var rail = document.getElementById("deckRail");
    var prev = document.getElementById("deckPrev");
    var next = document.getElementById("deckNext");
    var counter = document.getElementById("deckCounter");
    var idx = 0;

    panels.forEach(function (_, i) {
      var dot = document.createElement("button");
      dot.className = "figdeck-dot" + (i === 0 ? " is-active" : "");
      dot.type = "button";
      dot.setAttribute("aria-label", "Show figure " + (i + 1));
      dot.addEventListener("click", function () { show(i); });
      rail.appendChild(dot);
    });
    var dots = rail.querySelectorAll(".figdeck-dot");

    function show(i) {
      idx = Math.max(0, Math.min(panels.length - 1, i));
      panels.forEach(function (p, j) { p.classList.toggle("is-active", j === idx); });
      dots.forEach(function (d, j) { d.classList.toggle("is-active", j === idx); });
      counter.textContent = (idx + 1) + " / " + panels.length;
      prev.disabled = idx === 0;
      next.disabled = idx === panels.length - 1;
    }
    prev.addEventListener("click", function () { show(idx - 1); });
    next.addEventListener("click", function () { show(idx + 1); });
    deck.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") show(idx - 1);
      if (e.key === "ArrowRight") show(idx + 1);
    });
    deck.setAttribute("tabindex", "0");
    show(0);
  }

  /* ---------- demo suite filter ---------- */

  var pills = document.querySelectorAll(".filter-pill");
  pills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      pills.forEach(function (p) { p.classList.remove("active"); });
      pill.classList.add("active");
      var suite = pill.getAttribute("data-suite");
      document.querySelectorAll(".demo-card").forEach(function (card) {
        var showIt = suite === "all" || card.getAttribute("data-suite") === suite;
        card.classList.toggle("hidden", !showIt);
      });
    });
  });

  /* ---------- navbar active link + progress on scroll ---------- */

  var navLinks = document.querySelectorAll(".header-nav a");
  var sections = [];
  navLinks.forEach(function (a) {
    var id = a.getAttribute("href").slice(1);
    var el = document.getElementById(id);
    if (el) sections.push({ id: id, el: el, link: a });
  });

  function onScroll() {
    updateProgress();
    var pos = window.scrollY + 130;
    var current = sections[0] && sections[0].id;
    sections.forEach(function (s) { if (s.el.offsetTop <= pos) current = s.id; });
    navLinks.forEach(function (a) {
      a.classList.toggle("active", a.getAttribute("href") === "#" + current);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- reveal on scroll ---------- */

  if ("IntersectionObserver" in window && !reduceMotion) {
    var wio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); wio.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -40px 0px" });
    document.querySelectorAll(".reveal").forEach(function (el) { wio.observe(el); });
  } else {
    document.querySelectorAll(".reveal").forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- BibTeX copy ---------- */

  var btn = document.getElementById("copyBibtex");
  if (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent.trim();
      var done = function () {
        btn.textContent = "Copied ✓";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = "Copy BibTeX";
          btn.classList.remove("copied");
        }, 1800);
      };
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { /* noop */ }
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fallback);
      } else { fallback(); }
    });
  }
})();
