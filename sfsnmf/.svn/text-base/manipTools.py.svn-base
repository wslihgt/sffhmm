"""script with tools to manipulate files, directories, and so forth.

 copyright (C) 2010 - 2013 Jean-Louis Durrieu
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from os import listdir, path

def recursiveSearchFromRoot(rootDir,
                            conditionExtension='wav',
                            excludeExtension=['ces', 'cfs', 'chs', 'cgs',
                                              'trs', 'des']):
    """recursively explores `rootDir`, and lists the files with the given
    extension, and excluding the files with the provided list of extensions
    to avoid.
    """
    dirObj = listdir(rootDir)
    fileList = []
    numberOfFilesInRootDir = 0
    for k in dirObj:
        fullPath = rootDir+'/'+k
        if path.isdir(fullPath):
            #print "exploring ", fullPath
            fileList.extend(recursiveSearchFromRoot(fullPath,
                            conditionExtension=conditionExtension,
                            excludeExtension=excludeExtension))
        elif k[-len(conditionExtension):]==conditionExtension and \
                 not(k[-3:] in excludeExtension):
            fileList.append(fullPath)
            numberOfFilesInRootDir += 1
    
    print "    Stored ", numberOfFilesInRootDir, "files from ", rootDir
    return fileList

