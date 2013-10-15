#!/bin/bash

ls . | grep -v "$(basename $0)" | grep -v README | xargs rm -rf
