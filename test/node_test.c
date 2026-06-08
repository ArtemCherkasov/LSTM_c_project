//
// Created by User on 08.06.2026.
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include "../src/nn/common/node/node.h"

void assert_node(int inputCount, int sizeOfNode){
    struct Node node = createNode(inputCount);
    assert(node.inputCount == inputCount);
    assert(node.sizeOfNode == sizeOfNode);
    removeNode(node);
}

int main(){
    assert_node(220, 5352);
    return 0;
}