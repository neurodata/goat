
# Data descriptor for C. elegans connectome from Scott Emmons and co. 
### Ben Pedigo
### Mar 2019

## master_cell.csv 
Information for each cell in the following columns

    name : the name of the cell 
    type1 : broad category, in {endorgan, interneuron, motorneuron, sensory}
        endorgan is mostly muscle, other non-neuronal stuff
        the other categories are quite loose, and likely they truly overlap
        interneuron : something not predominantly motor or sensory
        motorneuron : strong projection to muscle 
        sensory : perceives something about the environment 
    type2 : more specific cell types within the type1 classes
        trust these even less than the above 
        many were categorized by some algorithm 
        some are useful for the endorgans, may specify whether 
        something is a muscle for example
    type3 : even more specific, only a few of these have data 
        might be things like subtype of muscle 
    whitetype : same level as type1, just different opinion
        from original White et al. papers in the 80s 
    sex : {'male', 'herm', 'both'}
        specifies sex that cell is present in, or both
    pharynx : {'phayrnx', 'nonpharynx', 'linker'}
        whether the cell is part of the pharynx network
        the pharynx is an almost isolated subnetwork, kinda like a mouth brain
        linker are the two cells that form the bridge between them
    sidepaired {'left', 'right', 'na'}: 
        'left' if cell is a l/r homolog and is on the left
        'right' if cell is a l/r homolog and is on the right
        'na' otherwise, may be on one side of the body but does not have a partner
    homolog : the name of another cell 
        if the cell has a homolog on the other side, this is the 
        name of that cell. 
    
## File organization and naming: 
Adjacency matrices are saved as csv, with the naming convention 
> `sex`\_`type`\_A\_`lcc`\_`directedness`.csv

`sex` is **male** or **herm**

`type` is **chem** or **gap**
 - **chem** is for 

`lcc` is **full**, **self**, or **multi**
 - **full** is the entire connectome of that type, may not be connected
 - **self** is the largest connected component of **full**
 - **multi** is the intersection of largest connected component with the opposite sex for that type

`directedness` is **undirected** or none
 - **undirected** for an symmetric view of the graph 
 - otherwise, assume the graph is directed 

The above adjacency matrices are indexed by cell names stored in 
> `sex`\_`type`\_`lcc`\_cells.csv

with the variable name portions defined as above

You can load the _cells.csv file for the corresponding adjacency matrix, and also load the master_cells.csv file to get metadata for those cells
# Other notes
For the gap junctions 
    used the "asymmetric" connectomes from Scott's connectome file
    manually removed entries on the lower diagonal that were nonzero 

