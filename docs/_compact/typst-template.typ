// =====================================================================
// Privify — Technical Analysis Document (COMPACT)
// Minimal, sober academic-paper template (arXiv/IEEE style).
//
// Monochrome (black only), no logo, no separate cover page: title, author,
// and abstract sit at the top of page 1 and the body starts immediately
// below. Dense paper spacing keeps the full content within the page budget.
//
// This partial is used ONLY by technical_analysis_compact.qmd. The long
// document keeps its branded template (docs/typst-template.typ).
//
// The `#let article(...)` signature is kept identical to Quarto's default
// partial so this file stays drop-in compatible with `typst-show.typ`.
// `content-to-string` comes from Quarto's `definitions.typ`, concatenated
// before this file.
// =====================================================================

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

  // ---------- Base typography (serif body, monochrome) ----------
  set text(font: "New Computer Modern", size: fontsize, fill: black, lang: lang, region: region)
  set par(justify: true, leading: linestretch * 0.6em, spacing: 0.6em, first-line-indent: 1.2em)
  show math.equation: set text(font: mathfont) if mathfont != none

  // ---------- Links (monochrome) ----------
  show link: set text(fill: black)

  // ---------- Section headings (bold serif, black, numbered, dense) ----------
  set heading(numbering: sectionnumbering)
  show heading: set text(fill: black)
  show heading.where(level: 1): it => block(above: 1.0em, below: 0.45em, width: 100%, {
    set text(size: 1.2em, weight: "bold")
    if it.numbering != none {
      numbering(it.numbering, ..counter(heading).at(it.location()))
      h(0.6em)
    }
    it.body
  })
  show heading.where(level: 2): it => block(above: 0.7em, below: 0.3em, width: 100%, {
    set text(size: 1.05em, weight: "bold")
    if it.numbering != none {
      numbering(it.numbering, ..counter(heading).at(it.location()))
      h(0.5em)
    }
    it.body
  })
  show heading.where(level: 3): it => block(above: 0.55em, below: 0.25em, width: 100%, {
    set text(size: 1.0em, weight: "bold", style: "italic")
    if it.numbering != none {
      numbering(it.numbering, ..counter(heading).at(it.location()))
      h(0.5em)
    }
    it.body
  })

  // ---------- Tables (booktabs-style, monochrome) ----------
  set table(stroke: none, inset: (x: 6pt, y: 4pt))
  show table: set text(size: 0.92em)
  show table.cell.where(y: 0): set text(weight: "bold")

  // ---------- Figure captions ----------
  show figure.caption: set text(size: 0.9em)

  // ---------- Page: centred page-number footer, no running header ----------
  set page(
    numbering: "1",
    footer: context align(center, text(size: 0.85em, counter(page).display("1"))),
  )

  // =====================================================================
  // TITLE BLOCK (paper style: no cover page, no logo, no colour)
  // =====================================================================
  align(center, {
    text(size: 1.7em, weight: "bold")[#title]
    if subtitle != none {
      v(0.45em)
      text(size: 1.15em, weight: "bold")[#subtitle]
    }
    v(0.9em)
    let authors-list = if authors == none { () } else { authors }
    for a in authors-list {
      text(size: 1.0em)[#a.name]
      let aff = a.at("affiliation", default: none)
      if aff != none {
        linebreak()
        text(size: 0.9em)[#aff]
      }
      let mail = a.at("email", default: none)
      if mail != none {
        linebreak()
        text(size: 0.85em)[#content-to-string(mail)]
      }
    }
    if date != none {
      v(0.4em)
      text(size: 0.9em)[#date]
    }
  })
  v(1.0em)

  // =====================================================================
  // ABSTRACT
  // =====================================================================
  if abstract != none {
    block(width: 100%, inset: (x: 1.6em), {
      align(center, text(weight: "bold")[
        #if abstract-title != none { abstract-title } else { "Abstract" }])
      v(0.3em)
      set par(first-line-indent: 0pt, justify: true)
      set text(size: 0.95em)
      abstract
    })
    v(0.6em)
  }

  // ---------- Keywords (paper style) ----------
  if keywords != none and keywords != () {
    block(width: 100%, inset: (x: 1.6em), {
      set text(size: 0.9em)
      [*Keywords:* #keywords.join(", ")]
    })
    v(0.7em)
  }

  // =====================================================================
  // TABLE OF CONTENTS (skipped when toc:false)
  // =====================================================================
  if toc {
    block(text(size: 1.2em, weight: "bold")[
      #if toc_title != none { toc_title } else { "Contents" }])
    v(0.4em)
    outline(title: none, depth: toc_depth, indent: toc_indent)
    v(0.7em)
  }

  // =====================================================================
  // BODY
  // =====================================================================
  doc
}
