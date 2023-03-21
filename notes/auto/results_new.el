(TeX-add-style-hook
 "results_new"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "../tables/relative_demand"
    "article"
    "art10"
    "booktabs"
    "caption"
    "geometry"))
 :latex)

