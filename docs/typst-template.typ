// =====================================================================
// Privify — Technical Analysis Document
// Custom Typst template (portfolio-grade), overriding Quarto's default
// `typst-template.typ` partial.
//
// The `#let article(...)` signature is kept identical to Quarto's default
// partial so this file stays drop-in compatible with the default
// `typst-show.typ` (no "unexpected argument" errors). Only the body is
// rewritten to produce the branded cover, abstract page, table of
// contents, running headers/footers, and section/code/table/caption
// styling.
//
// `content-to-string` is provided by Quarto's `definitions.typ` partial,
// which is concatenated before this file, so it is available here.
// =====================================================================

// ---------- Brand palette ----------
#let epimagenta = rgb("#D6006E")
#let epipurple  = rgb("#5B2A86")
#let epinavy    = rgb("#1C2541")
#let epiink     = rgb("#1A1A1A")
#let epigrey    = rgb("#6B7280")
#let epirule    = rgb("#E3E6EC")
#let epibg      = rgb("#F6F7F9")

// ---------- Fonts (self-contained: all bundled with Typst) ----------
// Body keeps New Computer Modern (serif). TeX Gyre Heros (the LaTeX sans)
// is not bundled with Typst, so DejaVu Sans is used as the self-contained
// sans substitute for titles, headers, table headers and captions.
#let SERIF = "New Computer Modern"
#let SANS  = "DejaVu Sans"
#let MONO  = "DejaVu Sans Mono"

// ---------- Fixed branding for this document ----------
#let DOC_KIND      = "Technical Analysis Document"
#let EXAM_LINE     = "Final Examination Project"
#let COURSE        = "Computer Vision"
#let ACADEMIC_YEAR = "Academic Year 2025/2026"
#let REPO_URL      = "https://github.com/jenz26/privify"
#let REPO_LABEL    = "github.com/jenz26/privify"

// Mid-dot separator (magenta) used on the cover.
#let midsep = [#h(0.55em)#text(fill: epimagenta, weight: "bold")[·]#h(0.55em)]

// ---------- Code listing container override ----------
// Quarto emits highlighted code through a `Skylighting(...)` helper whose
// background colour is hard-coded and which draws no border. This file is
// concatenated after Quarto's definitions, so redefining the helper here
// shadows the original and gives the branded background + thin border.
// The signature mirrors Quarto 1.9's helper; token colours stay Quarto's
// default Python highlighting.
#let Skylighting(fill: none, number: false, start: 1, sourcelines) = {
  let blocks = []
  let lnum = start - 1
  for ln in sourcelines {
    if number {
      lnum = lnum + 1
      blocks = blocks + box(
        width: if start + sourcelines.len() > 999 { 30pt } else { 24pt },
        text(fill: epigrey, [ #lnum ]),
      )
    }
    blocks = blocks + ln + EndLine()
  }
  block(
    fill: epibg,
    stroke: 0.7pt + epirule,
    width: 100%,
    inset: 8pt,
    radius: 2pt,
    blocks,
  )
}

#let article(
  title: none,
  subtitle: none,
  authors: none,
  keywords: (),
  date: none,
  abstract-title: none,
  abstract: none,
  thanks: none,
  cols: 1,
  lang: "en",
  region: "US",
  font: none,
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: none,
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  mathfont: none,
  codefont: none,
  linestretch: 1,
  sectionnumbering: none,
  linkcolor: none,
  citecolor: none,
  filecolor: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  // ---------- PDF metadata ----------
  set document(title: title, keywords: keywords)
  set document(
    author: authors.map(a => content-to-string(a.name)).join(", ", last: " & "),
  ) if authors != none and authors != ()

  // ---------- Base typography ----------
  set text(font: SERIF, size: fontsize, fill: epiink, lang: lang, region: region)
  set par(
    justify: true,
    leading: linestretch * 0.65em,
    spacing: 0.95em,
    first-line-indent: 1.15em,
  )
  show math.equation: set text(font: mathfont) if mathfont != none

  // ---------- Links & cross-references ----------
  // URLs (string destinations) in purple; internal links (TOC, refs) in navy.
  show link: it => {
    let purple-link = type(it.dest) == str
    text(fill: if purple-link { epipurple } else { epinavy }, it)
  }
  show ref: set text(fill: epimagenta)

  // ---------- Section headings ----------
  set heading(numbering: sectionnumbering)
  show heading.where(level: 1): it => block(above: 1.6em, below: 0.7em, width: 100%, {
    set text(font: SANS, size: 1.4em, weight: "bold", fill: epinavy)
    if it.numbering != none {
      text(fill: epimagenta, numbering(it.numbering, ..counter(heading).at(it.location())))
      h(0.7em)
    }
    it.body
  })
  show heading.where(level: 2): it => block(above: 1.1em, below: 0.45em, width: 100%, {
    set text(font: SANS, size: 1.15em, weight: "bold", fill: epinavy)
    if it.numbering != none {
      text(fill: epimagenta, numbering(it.numbering, ..counter(heading).at(it.location())))
      h(0.6em)
    }
    it.body
  })
  show heading.where(level: 3): it => block(above: 0.9em, below: 0.4em, width: 100%, {
    set text(font: SANS, size: 1.0em, weight: "bold", fill: epinavy)
    if it.numbering != none {
      text(fill: epimagenta, numbering(it.numbering, ..counter(heading).at(it.location())))
      h(0.5em)
    }
    it.body
  })

  // ---------- Code listings ----------
  // Typst's native syntax highlighting handles Python tokens; here we only
  // style the container (light background, thin border) and the mono font.
  show raw: set text(font: MONO, size: 0.92em)
  show raw.where(block: true): set block(
    fill: epibg,
    stroke: 0.7pt + epirule,
    inset: 8pt,
    radius: 2pt,
    width: 100%,
  )

  // ---------- Tables (booktabs-style) ----------
  // Horizontal rules come from Pandoc's emitted table.hline() calls; the
  // grid itself has no strokes (no vertical lines). Header row is sans/navy.
  set table(stroke: none, inset: (x: 8pt, y: 5pt))
  // Smaller, non-hyphenated text keeps narrow label columns from breaking
  // mid-word (e.g. "Precision") when the data columns carry long headers.
  show table: set text(size: 0.9em, hyphenate: false)
  show table.cell.where(y: 0): set text(font: SANS, weight: "bold", fill: epinavy)

  // ---------- Figure / table captions ----------
  show figure.caption: it => block(width: 100%, {
    set text(size: 0.92em)
    text(font: SANS, weight: "bold", fill: epinavy)[
      #it.supplement #context it.counter.display(it.numbering)#it.separator]
    emph(it.body)
  })

  // ---------- Author data (driven by metadata) ----------
  let primary = if authors != none and authors.len() > 0 { authors.first() } else { none }
  let author-name  = if primary != none { primary.name } else { [Marco Contin] }
  let author-aff   = if primary != none { primary.affiliation } else { [Epicode Institute of Technology] }
  let author-email = if primary != none { content-to-string(primary.email) } else { "" }
  let author-sid   = if author-email.contains("@") { author-email.split("@").first() } else { "" }

  // =====================================================================
  // COVER PAGE — suppress header, footer and page number (Quarto's page.typ
  // sets a global numbering, so it must be explicitly cleared here).
  // =====================================================================
  set page(header: none, footer: none, numbering: none)
  {
    set align(center)

    line(length: 100%, stroke: 2pt + epimagenta)
    v(2pt)
    line(length: 100%, stroke: 0.6pt + epinavy)

    v(1.5cm)
    image("/assets/logo_epicode_dark.png", height: 2cm)

    v(1.1cm)
    text(font: SANS, size: 1.1em, fill: epigrey)[#EXAM_LINE]
    v(0.35em)
    text(font: SANS, size: 1.1em, weight: "bold", fill: epinavy)[#COURSE]
    v(0.25em)
    text(font: SANS, fill: epigrey)[#author-aff]

    v(1fr)

    line(length: 32%, stroke: 0.8pt + epirule)
    v(0.85cm)
    text(font: SANS, size: 0.85em, weight: "bold", fill: epimagenta)[#upper(DOC_KIND)]
    v(0.7cm)
    text(size: 52pt, weight: "bold", fill: epinavy)[#title]
    v(0.55cm)
    if subtitle != none {
      text(size: 1.2em, style: "italic")[#subtitle]
    }
    v(0.85cm)
    line(length: 32%, stroke: 0.8pt + epirule)

    v(1fr)

    block(width: 85%, {
      set align(center)
      text(font: SANS, size: 0.85em, fill: epigrey)[#upper("Author")]
      v(0.3em)
      text(size: 1.2em)[#author-name]
      v(0.3em)
      if author-sid != "" {
        text(size: 0.9em, fill: epigrey)[Student ID #text(weight: "bold", fill: epiink)[#author-sid]#midsep#author-email]
      } else {
        text(size: 0.9em, fill: epigrey)[#author-email]
      }
    })

    v(1fr)

    line(length: 100%, stroke: 0.6pt + epinavy)
    v(2pt)
    line(length: 100%, stroke: 2pt + epimagenta)
    v(0.35cm)
    text(font: SANS, size: 0.85em, fill: epigrey)[
      #ACADEMIC_YEAR#midsep#date#midsep#link(REPO_URL)[#REPO_LABEL]]
  }
  pagebreak()

  // =====================================================================
  // Running header + page-number footer for every page after the cover.
  // =====================================================================
  set page(
    numbering: "1",
    header: context {
      let above = query(selector(heading.where(level: 1)).before(here()))
      let current = if above.len() > 0 { above.last().body } else { [] }
      set text(font: SANS, size: 0.78em, fill: epigrey)
      grid(
        columns: (1fr, auto),
        align(left)[Privify #text(fill: epimagenta, weight: "bold")[/] #DOC_KIND],
        align(right)[#current],
      )
      v(-0.55em)
      line(length: 100%, stroke: 0.6pt + epirule)
    },
    footer: context {
      set text(font: SANS, size: 0.78em, fill: epigrey)
      align(center)[#counter(page).display("1")]
    },
  )
  counter(page).update(1)

  // =====================================================================
  // ABSTRACT
  // =====================================================================
  if abstract != none {
    v(0.5cm)
    align(center, {
      text(font: SANS, size: 1.4em, weight: "bold", fill: epinavy)[
        #if abstract-title != none { abstract-title } else { "Abstract" }]
      v(0.3em)
      line(length: 2.4cm, stroke: 1.5pt + epimagenta)
    })
    v(0.7cm)
    block(inset: (x: 1.2em), {
      set text(style: "italic")
      set par(first-line-indent: 0pt, justify: true)
      abstract
    })
    pagebreak()
  }

  // =====================================================================
  // TABLE OF CONTENTS
  // =====================================================================
  if toc {
    block({
      text(font: SANS, size: 1.4em, weight: "bold", fill: epinavy)[
        #if toc_title != none { toc_title } else { "Contents" }]
      v(0.2em)
      line(length: 2.4cm, stroke: 1.5pt + epimagenta)
    })
    v(0.7cm)
    show outline.entry.where(level: 1): it => strong(it)
    show outline.entry.where(level: 2): it => emph(it)
    outline(title: none, depth: toc_depth, indent: toc_indent)
    pagebreak()
  }

  // =====================================================================
  // BODY
  // =====================================================================
  doc
}
