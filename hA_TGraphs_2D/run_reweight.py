from hA2025Reweighter import hA2025Reweighter

rw = hA2025Reweighter(
    "TGraphs_2018.root",
    "TGraphs_2025.root"
)

# Example
rw.print_info(    KE=100,  fate="cex",  A=40 )

#rw.debug2018(100, 40)