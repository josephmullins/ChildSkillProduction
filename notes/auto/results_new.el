(TeX-add-style-hook
 "results_new"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "output"
    "article"
    "art10"
    "booktabs"
    "caption"))
 :latex)

