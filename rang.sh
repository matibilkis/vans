for (( k = 0; k < 10; ++k )) do
  a=$(( echo print(k/10) | python3 ))
  echo "$a"
done
